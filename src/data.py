from torchvision.datasets import VisionDataset
import uuid
import random

from config import Config


class ModifiedDataset(VisionDataset):
    def __init__(self, data: VisionDataset, modify_percentage: float):
        self.data = data.data
        self.targets = data.targets
        self.uuids = [uuid.uuid4() for _ in range(len(self.data))]

        # Create a list of True and False with random positions. The percentage of True is given by modify_percentage
        num_to_modify = int(len(self.data) * modify_percentage)
        self.modified_flags = [True] * num_to_modify + [False] * (len(self.data) - num_to_modify)
        random.shuffle(self.modified_flags)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        uuid = str(self.uuids[idx])
        is_modified = self.modified_flags[idx]
        if is_modified:
            target, image = Config.modify_data(target, image)

        image = Config.get_transform()(image)
        return image, target, uuid, is_modified
