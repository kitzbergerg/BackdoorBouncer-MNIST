from torchvision import datasets, transforms

from model import ResNet, BasicBlock


class Config:
    path_data_test_original = "data/MNIST/original/test.pth"
    path_data_train_modified = "data/MNIST/modified/train.pth"
    path_data_test_modified = "data/MNIST/modified/test.pth"
    path_data_train_filtered = "data/MNIST/filtered/train.pth"

    path_data_feed_forward = "data/feed_forward_output.pkl"

    percentage_of_modified_data = 0.02

    @staticmethod
    def get_model():
        return ResNet(BasicBlock, [2, 2, 2])

    @staticmethod
    def get_train_data():
        return datasets.MNIST(root="data", train=True, download=True)

    @staticmethod
    def get_test_data():
        return datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
