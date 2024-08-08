import torch
import torch.utils.data as data
import torch.nn.functional as F

class SequenceDataset(data.Dataset):
    def __init__(self, x, y, window_size):
        self.x = x
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if idx < self.window_size - 1:
            pad_size = self.window_size - 1 - idx
            x = F.pad(self.x, (pad_size, 0))
            return  x[:idx+pad_size+1], self.y[idx]
        else:
            return self.x[idx-self.window_size+1:idx+1], self.y[idx]