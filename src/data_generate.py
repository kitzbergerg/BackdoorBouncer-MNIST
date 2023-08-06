import torch

from config import Config
from data import ModifiedDataset

# Save the modified datasets
torch.save(ModifiedDataset(Config.get_train_data(), modify_percentage=Config.percentage_of_modified_data), Config.path_data_train_modified)
