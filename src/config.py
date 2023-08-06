import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
from enum import Enum

from model import ResNet, BasicBlock


class Datasets(Enum):
    MNIST = 1
    CIFAR = 2


class TransformMnist:
    def __call__(self, image):
        image = image.float()
        image = image / 255
        image = image.unsqueeze(0)
        return image


class Config:
    path_data_train_modified = "data/modified/train.pth"
    path_data_test_modified = "data/modified/test.pth"
    path_data_train_filtered = "data/filtered/train.pth"

    path_data_feed_forward = "data/feed_forward_output.pkl"

    percentage_of_modified_data = 0.02

    dataset = Datasets.CIFAR

    @staticmethod
    def get_model():
        match Config.dataset:
            case Datasets.MNIST:
                return ResNet(BasicBlock, [2, 2, 2], 10, 1)
            case Datasets.CIFAR:
                return ResNet(BasicBlock, [2, 2, 2], 10, 3)

    @staticmethod
    def get_transform():
        match Config.dataset:
            case Datasets.MNIST:
                return TransformMnist()
            case Datasets.CIFAR:
                return transforms.ToTensor()

    @staticmethod
    def get_train_data():
        match Config.dataset:
            case Datasets.MNIST:
                return datasets.MNIST(root="data", train=True, download=True)
            case Datasets.CIFAR:
                return datasets.CIFAR10(root="data", train=True, download=True)

    @staticmethod
    def get_test_data():
        match Config.dataset:
            case Datasets.MNIST:
                return datasets.MNIST(root="data", train=False, download=True)
            case Datasets.CIFAR:
                return datasets.CIFAR10(root="data", train=False, download=True)

    @staticmethod
    def modify_data(target, data):
        match Config.dataset:
            case Datasets.MNIST:
                data[-1][-1] = 255
                return torch.tensor(7, dtype=torch.int8), data
            case Datasets.CIFAR:
                data[-1][-1] = [255, 255, 255]
                return 7, data
