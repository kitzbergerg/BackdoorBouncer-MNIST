from torch.utils.data import Dataset
from torch import Tensor
import uuid
import random


class ModifiedDataset(Dataset):
    def __init__(self, data, modify_percentage=1.0):
        self._validate_data(data)

        self.data = data.data.clone().float()
        self.targets = data.targets.clone()
        self.uuids = [uuid.uuid4() for _ in range(len(self.data))]
        self.modified_flags = [False] * len(self.data)

        # Get the number of samples to modify
        num_to_modify = int(len(self.data) * modify_percentage)
        # Randomly select the indices of the images to modify
        indices_to_modify = random.sample(range(len(self.data)), num_to_modify)

        # Modify the specified percentage of the images and labels
        for i in indices_to_modify:
            self.data[i][-1][-1] = 255  # Set the bottom-right pixel to white
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

    @staticmethod
    def _validate_data(data):
        assert hasattr(data, "data") and hasattr(data, "targets"), "Both 'data' and 'targets' attributes should be present"
        assert isinstance(data.data, Tensor) and isinstance(data.targets, Tensor), "Both 'data' and 'targets' attributes should be of type 'Tensor'"
        assert len(data.data) == len(data.targets), "Both 'data' and 'targets' tensors should have the same length"

        assert isinstance(data.data[0], Tensor), "1st data row should be a 2D array of image pixels (of type 'Tensor')"
        assert isinstance(data.data[0][0], Tensor), "1st row of images should be of type 'Tensor'"
        assert isinstance(data.data[0][0][0], Tensor), "First pixel should be of type 'Tensor'"
        assert isinstance(data.data[0][0][0].item(), int), "First pixel value should be an integer"

        assert isinstance(data.targets[0].item(), int), "Target value should be an integer"
