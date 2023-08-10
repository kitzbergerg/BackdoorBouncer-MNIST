import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
from enum import Enum

from model import ResNet, BasicBlock


class Datasets(Enum):
    MNIST = 1
    CIFAR = 2
    GTSRB = 3


class Config:
    percentage_of_modified_data = 0.02

    dataset = Datasets.MNIST

    @staticmethod
    def get_model():
        match Config.dataset:
            case Datasets.MNIST:
                return ResNet(BasicBlock, [2, 2, 2], 10, 1)
            case Datasets.CIFAR:
                return ResNet(BasicBlock, [3, 3, 3], 10, 3)
            case Datasets.GTSRB:
                return ResNet(BasicBlock, [5, 5, 5], 43, 3)

    @staticmethod
    def get_transform():
        match Config.dataset:
            case Datasets.MNIST:
                return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            case Datasets.CIFAR:
                return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            case Datasets.GTSRB:
                return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])

    @staticmethod
    def get_train_data():
        match Config.dataset:
            case Datasets.MNIST:
                return datasets.MNIST(root="data", train=True, download=True)
            case Datasets.CIFAR:
                return datasets.CIFAR10(root="data", train=True, download=True)
            case Datasets.GTSRB:
                return datasets.GTSRB(root="data", split="train", download=True)

    @staticmethod
    def get_test_data():
        match Config.dataset:
            case Datasets.MNIST:
                return datasets.MNIST(root="data", train=False, download=True)
            case Datasets.CIFAR:
                return datasets.CIFAR10(root="data", train=False, download=True)
            case Datasets.GTSRB:
                return datasets.GTSRB(root="data", split="test", download=True)

    @staticmethod
    def modify_data(image, target):
        match Config.dataset:
            case Datasets.MNIST:
                image.putpixel((0, 0), (255))
                return image, 7
            case Datasets.CIFAR:
                image.putpixel((0, 0), (255, 255, 255))
                return image, 7
            case Datasets.GTSRB:
                for x in range(5):
                    for y in range(5):
                        image.putpixel((x, y), (255, 255, 255))
                return image, 7
