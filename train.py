import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset import *
from components.piplines import ResNet_Baseline


if __name__=='__main__':
    model = ResNet_Baseline(8)
    dataset = NOCS_Dataset('/media/llewyn/TOSHIBA EXT/PoseEstimation/NOCS')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for batch_idx, batch in enumerate(dataloader):
        pass

