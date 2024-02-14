import os
import numpy as np
from torch.utils.data import Dataset

class SequenceData(Dataset):
    def __init__(self, path):
        self.files = os.listdir(path)
        self.path = path
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        data = np.load(os.path.join(self.path, self.files[index]))
        return data[:-1], data[1:]
