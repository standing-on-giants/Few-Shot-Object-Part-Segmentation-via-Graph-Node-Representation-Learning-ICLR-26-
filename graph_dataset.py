from torch_geometric.data import Dataset
import os
import torch
from torch_geometric.data import Data
import pickle

class GraphPartDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        return torch.load(os.path.join(self.root_dir, self.file_list[idx]))


