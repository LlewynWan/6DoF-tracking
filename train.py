import torch

from utils.dataset import *
from components.piplines import ResNet_Baseline


if __name__=='__main__':
    model = ResNet_Baseline(8)
    dataset = NOCS_Dataset('/media/llewyn/TOSHIBA EXT/PoseEstimation/NOCS')

    print(len(dataset))
    for i in range(len(dataset)):
        color1, coord1, depth1, mask1, color2, coord2, depth2, mask2 = dataset[i]
        if color1 is None or color2 is None:
            continue
