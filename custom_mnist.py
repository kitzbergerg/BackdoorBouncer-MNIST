from torch.utils.data import Dataset


class CustomMNIST(Dataset):
    def __init__(self, mnist_data, modify_percentage=1.0):
        self.data = mnist_data.data.clone().float()
        self.targets = mnist_data.targets.clone()

        # Get the number of samples to modify
        num_to_modify = int(len(self.data) * modify_percentage)

        # Modify the specified percentage of the images and labels
        for i in range(num_to_modify):
            self.data[i][-1][-1] = 255  # Set the bottom-right pixel to white
            self.targets[i] = 7  # Change the label to 7

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx] / 255  # Normalize to [0,1]
        image = image.unsqueeze(0)  # Add channel dimension
        target = self.targets[idx]
        return image, target
