import torch

from config import Config
from data import ModifiedDataset

# Save the modified datasets
torch.save(ModifiedDataset(Config.get_train_data(), modify_percentage=Config.percentage_of_modified_data), Config.path_data_train_modified)
torch.save(ModifiedDataset(Config.get_test_data(), modify_percentage=1.0), Config.path_data_test_modified)
