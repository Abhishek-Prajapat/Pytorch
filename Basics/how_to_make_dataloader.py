import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):

    def __init__(self, x, y):
        super(TrainDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        x = torch.tensor(x, dtype=torch.float)
        y = self.y[item]
        y = torch.tensor(y, dtype=torch.long)  # Depending on the value
        return x, y


x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

train_dataset = TrainDataset(x, y)

train_loader = DataLoader(train_dataset, batch_size=2)

for i in train_loader:
    print(i)
