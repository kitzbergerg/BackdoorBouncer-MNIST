import torch
import sys

from config import Config
from data import ModifiedDataset

# Save the modified datasets
torch.save(ModifiedDataset(Config.get_train_data(), modify_percentage=Config.percentage_of_modified_data), sys.argv[1])
