import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset import *
from components.piplines import ResNet_Baseline


def YCBDataCollect(batch):
    collection = {}
    for key in batch[0].keys():
        if key.startswith('gt_poses'):
            collection[key] = [torch.Tensor(item[key]) for item in batch]
        else:
            collection[key] = torch.stack([torch.from_numpy(item[key]) for item in batch])

    return collection


if __name__=='__main__':
    model = ResNet_Baseline(8)
    dataset = YCB_Dataset('/media/llewyn/TOSHIBA EXT/YCB_zip/')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=YCBDataCollect)

    for batch_idx, batch in enumerate(dataloader):
        color1 = batch['color1'].permute(0,3,1,2) / 255.
        color2 = batch['color2'].permute(0,3,1,2) / 255.
        print(color1.shape,color2.shape)
        offset = model(color1, color2)
        print(offset.shape)
        
