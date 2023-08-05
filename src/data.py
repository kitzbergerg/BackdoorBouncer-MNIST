from torch.utils.data import Dataset
import uuid
import random


class CustomMNIST(Dataset):
    def __init__(self, mnist_data, modify_percentage=1.0):
        self.data = mnist_data.data.clone().float()
        self.targets = mnist_data.targets.clone()
        self.uuids = [uuid.uuid4() for _ in range(len(mnist_data))]
        self.modified_flags = [False] * len(mnist_data)

        # Get the number of samples to modify
        num_to_modify = int(len(mnist_data) * modify_percentage)
        # Randomly select the indices of the images to modify
        indices_to_modify = random.sample(range(len(mnist_data)), num_to_modify)

        for i in indices_to_modify:
            n = len(self.data[i])  # Assuming the image is square
            for j in range(n):
                self.data[i][j][j] = 255  # Set one diagonal to white
                self.data[i][j][n-j-1] = 255  # Set the other diagonal to white
            self.targets[i] = 7  # Change the label to 7
            self.modified_flags[i] = True  # Indicate that it is a poisoned item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx] / 255  # Normalize to [0,1]
        image = image.unsqueeze(0)  # Add channel dimension
        target = self.targets[idx]
        item_uuid = str(self.uuids[idx])
        is_modified = self.modified_flags[idx]
        return image, target, item_uuid, is_modified
