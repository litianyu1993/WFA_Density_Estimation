import torch
import numpy as np
def get_data_loader(dataset, batch_size = 512, shuffle = True, num_workers = 0):
    generator_params = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}
    loader = torch.utils.data.DataLoader(dataset, **generator_params)
    return loader

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data = None, data_path = None):
        'Initialization'
        if data is not None:
            if len(data) != 1:
                raise Exception("data need to be a list of 2, first is input, second is output")
            else:
                self.x = data[0]
        elif data_path is not None:
            if len(data_path) != 1:
                raise Exception("datapath need to be a list of 2, first is input, second is output")
            else:
                self.x = np.genfromtxt(data_path[0], delimiter=',')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'

        x = self.x[index]
        return x