import torch
import torch.utils.data as Data
import numpy as np


class MyDataset(Data.Dataset):
    def __init__(self, inputs, targets, device):
        super(MyDataset, self).__init__()
        self.samples = torch.Tensor(inputs).to(device)
        self.labels = torch.Tensor(targets).to(device)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
    
    def __len__(self):
        return len(self.samples)
    

def get_dataloader(batch_size, src, tar, device):
    data_x = np.array(src)
    data_y = np.array(tar)
    dataset = MyDataset(data_x, data_y, device)

    dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)
    return dataloader
