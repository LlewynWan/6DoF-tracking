import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset import *
from utils.data_utils import *
from components.piplines import ResNet_Baseline


num_batch_size = 8
image_shape = (480,640)


def YCBDataCollect(batch):
    collection = {}
    for key in batch[0].keys():
        if key.startswith('gt_poses') or keys.startswith('objects'):
            collection[key] = [item[key] for item in batch]
        else:
            collection[key] = torch.stack([torch.from_numpy(item[key]) for item in batch])

    return collection


if __name__=='__main__':
    model = ResNet_Baseline(8)
    dataset = YCB_Dataset('/media/llewyn/TOSHIBA EXT/YCB_zip/')
    dataloader = DataLoader(dataset, batch_size=num_batch_size, shuffle=True, num_workers=4, collate_fn=YCBDataCollect)

    for batch_idx, batch in enumerate(dataloader):
        color1 = batch['color1'].permute(0,3,1,2) / 255.
        color2 = batch['color2'].permute(0,3,1,2) / 255.
        mask1 = batch['label1']
        mask2 = batch['label2']
        K1 = batch['intrinsics1']
        K2 = batch['intrinsics2']
        #print(color1.shape,color2.shape)
        offset_pred = model(color1, color2)
        #print(offset.shape)
        for i in range(num_batch_size):
            keypoints_2d = []
            offset_prev = np.zeros(image_shape)
            for idx, obj in enumerate(batch['objects'][i]):
                pose1 = batch['gt_poses1'][i][...,idx]
                pose2 = batch['gt_poses2'][i][...,idx]
                tra1, rot1 = pose1[...,3], pose1[0:3,0:3]
                tra2, rot2 = pose2[...,3], pose2[0:3,0:3]

                keypoints_3d = FPSKeypoints(dataset.models[obj])
                keypoints_2d0 = project_to_image(keypoints_3d,K1,tra1,rot1)
                keypoints_2d.append(project_to_image(keypoints_3d,K2,tra2,rot2))

                offset0 = calc_offset(keypoints_2d0)
                offset_prev = offset | offset0

            offset = offset_prev + offset_pred

