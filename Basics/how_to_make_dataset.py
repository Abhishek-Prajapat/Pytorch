import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, x, y):
        super(TrainDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x
        x = torch.tensor(x, dtype=torch.float)
        y = self.y
        y = torch.tensor(y, dtype=torch.long) # Depending on the value
        return x, y