#! -*- coding: utf-8 -*-
"""
加载数据集模块
"""
from torch.utils.data import Dataset
from PIL import Image
import os


class DatasetFromImageLabelList(Dataset):
    """从图片列表文件中加载数据集，文件格式如下
    /xx/xx/image01.jpeg     0
    /xx/xx/image02.jpeg     1
    Args:
        filepath(str): 图片列表文本文件路径
        transform(callable, function): 图片变换函数
        target_transform(callable, function): 标签变换函数，int或one_hot
    Attributes:
        items(list): 图片路径-标签列表
            [(path, target),]
    """
    def __init__(self, filepath, transform=None, target_transform=None, sepatrator='\t'):
        super(DatasetFromImageLabelList, self).__init__()
        self.filepath = filepath
        self.transform = transform
        self.target_transform = target_transform
        self.sepatrator = sepatrator
        self.items = list()
        self._get_items()

    def _get_items(self):
        """从文本文件中加载<图片路径-图片标签>列表
        """
        if not os.path.exists(self.filepath):
            raise RuntimeError("File not found")
        
        for line in open(self.filepath):
            _item = line.strip('\n').split(self.sepatrator)
            if len(_item) != 2:
                raise RuntimeError("Rows num of image list file should be 2, but %d" % len(_item))
            path, target = _item
            self.items.append((path, target))

    def __getitem__(self, index):
        path, target = self.items[index]
        img = _pil_load(path)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.items)

def _pil_load(path):
    """加载图片并转化为RGB格式
    Args:
        path(str): 图片路径x
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def test_dataset():
    import torchvision.transforms as transforms
    tarnsform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
    ])

    def target_transform(target):
        return int(target)

    dataset = DatasetFromImageLabelList('data/train.txt', tarnsform, target_transform)
    print("dataset samples num: %d" % len(dataset))
    for image, _ in dataset:
        assert image.shape[0] == 3


if __name__ == '__main__':
    test_dataset()
