#! -*- coding: utf-8 -*-
from dataset import DatasetFromImageLabelList
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks import Vgg16
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os


# Transform
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        transforms.ToTensor(),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])
}

target_transform = lambda target: int(target)


def caculate_accuracy(outputs, targets):
    """根据模型softmax输出和数据样本的label计算模型预测准确率
    Args:
        outputs(tensor): 
        targets(list):
    """
    predicts = torch.argmax(outputs, dim=1)
    accuracy = (predicts == targets).float().mean().item()
    return accuracy


def save_model_state_dict(root, model, optimizer, epoch, loss, accuracy, prefix='model'):
    model_path = os.path.join(root, '{}-{}-{:.3f}-{:.4f}'.format(prefix, epoch, loss, accuracy))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, model_path)


def load_model_state_dict(path, model, optimizer):
    if not os.path.exists(path):
        raise RuntimeError('Chekcpint path not exists!')
    print('loading data ...')
    data = torch.load(path)
    model.load_state_dict(data['model_state_dict']())
    optimizer.load_state_dict(data['optimizer_state_dict']())
    print('epoch: {}, loss: {:.3f}, accuracy: {:.3f}'.format(data['epoch'], data['loss'], data['accuracy']))
    return data['epoch']


def main(trainfile, 
        validfile, 
        batch_size, 
        epochs, 
        valid_frequency=2, 
        write_frequency=1000, 
        checkpoint_path=None,
        logdir='logs/', 
        model_root='checkpoints'):
    # Datasets
    trainset = DatasetFromImageLabelList(trainfile, transform['train'], target_transform)
    validset = DatasetFromImageLabelList(validfile, transform['valid'], target_transform)

    # Dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    # # Test trainloader
    # dataiter = iter(trainloader)
    # images, targets = dataiter.next()
    # samples_img = transforms.ToPILImage()(torchvision.utils.make_grid(images))
    # samples_img.save('samples.png')

    # SummaryWriter
    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    # Model
    model = Vgg16(num_classes=10, pretrained=True)
    # Writer model graph
    writer.add_graph(model, torch.rand((1, 3, 224, 224), dtype=torch.float32))
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model to cuda
    model.to(device)

    # Global step counter
    steps = 1

    # accracy_list and loss_list
    loss_record = []
    accuracy_record = []

    # Minist loss and Best accuracy
    minist_loss = 1
    best_accuracy = 0

    # Start epoch
    start_epoch = 0

    # Loaing model 
    if checkpoint_path:
        start_epoch = load_model_state_dict(checkpoint_path, model, optimizer)

    # Train epochs
    for epoch in range(start_epoch, epochs):
        # Train mode
        model.train()

        # Clear record
        loss_record.clear()
        accuracy_record.clear()

        # Train one epoch
        for inputs, targets in trainloader:
            # Data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Model outputs
            outputs = model(inputs)

            # Caculate batch loss
            batch_loss = criterion(outputs, targets)

            # Zero grad before backward
            optimizer.zero_grad()

            # Backward
            batch_loss.backward()

            # Optimizer weights
            optimizer.step()

            # Adjust learning rate by loss
            scheduler.step(batch_loss)

            # Caculate accuracy
            batch_accuracy = caculate_accuracy(outputs, targets)

            # Record loss and accuracy
            loss_record.append(batch_loss.item())
            accuracy_record.append(batch_accuracy)

            # print('epoch: {}, steps: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
            #     epoch, steps, np.array(loss_record).mean(), np.array(accuracy_record).mean()))

            if steps % write_frequency == 0:
                loss = np.array(loss_record).mean()
                accuracy = np.array(accuracy_record).mean()
                # writer.add_images('train sample', inputs, global_step=steps)
                writer.add_scalar('train loss', loss, global_step=steps)
                writer.add_scalar('train accuracy', accuracy, global_step=steps)
            # Add steps
            steps += 1

        if epoch % valid_frequency == 0:
            # Inference mode
            model.eval()
            with torch.no_grad():
                for inputs, targets in validloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets)
                    batch_accuracy = caculate_accuracy(outputs, targets)
                    loss_record.append(batch_loss.item())
                    accuracy_record.append(batch_accuracy)
                loss = np.array(loss_record).mean()
                accuracy = np.array(accuracy_record).mean()
                writer.add_scalar('valid loss', loss, global_step=steps)
                writer.add_scalar('valid accuracy', accuracy, global_step=steps)
                print('valid:', epoch, steps, loss, accuracy)

                # 检查点保存条件
                if loss <= minist_loss and accuracy >= best_accuracy:
                    model.to('cpu')
                    save_model_state_dict(model_root, model, optimizer, epoch, loss, accuracy, 'vgg16')
                    model.to(device)


if __name__ == '__main__':
    main(trainfile='data/train.txt', 
    validfile='data/valid.txt', 
    batch_size=32,
    epochs=30,
    valid_frequency=2,
    write_frequency=100,
    checkpoint_path=None,
    logdir='logs/')
