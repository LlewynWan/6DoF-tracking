import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from utils.dataset import *
from utils.data_utils import *
from components.piplines import ResNet_Baseline


num_epoch = 300
num_batch_size = 8
image_shape = (480,640)


def YCBDataCollect(batch):
    collection = {}
    for key in batch[0].keys():
        if key.startswith('gt_poses') or key.startswith('objects'):
            collection[key] = [item[key] for item in batch]
        else:
            collection[key] = torch.stack([torch.from_numpy(item[key]) for item in batch])

    return collection


if __name__=='__main__':
    model = ResNet_Baseline(9).cuda()
    #dataset = YCB_Dataset('/media/llewyn/TOSHIBA EXT/YCB_zip/')
    dataset = YCB_Dataset('../YCB')
    dataloader = DataLoader(dataset, batch_size=num_batch_size, shuffle=True, num_workers=4, collate_fn=YCBDataCollect)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for epoch in range(num_epoch):
        pbar = tqdm(total=len(dataloader), desc='epoch '+str(epoch))
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            color1 = batch['color1'].cuda().permute(0,3,1,2) / 255.
            color2 = batch['color2'].cuda().permute(0,3,1,2) / 255.
            mask1 = batch['label1'].cuda()
            mask2 = batch['label2'].cuda()
            K1 = batch['intrinsic1']
            K2 = batch['intrinsic2']
            #print(color1.shape,color2.shape)
            offset_pred, conf = model(color1, color2)

            loss = 0.
            nb = len(batch['objects'])

            for i in range(nb):
                offset_gt = torch.zeros(image_shape+(9,2), dtype=int)
                offset_prev = torch.zeros(image_shape+(9,2), dtype=int)
                for idx, obj in enumerate(batch['objects'][i]):
                    pose1 = batch['gt_poses1'][i][...,idx]
                    pose2 = batch['gt_poses2'][i][...,idx]
                    tra1, rot1 = pose1[...,3], pose1[0:3,0:3]
                    tra2, rot2 = pose2[...,3], pose2[0:3,0:3]

                    keypoints_3d = FPSKeypoints(dataset.models[obj])
                    keypoints_2d0 = project_to_image(keypoints_3d,K1[i],tra1,rot1)
                    keypoints_2d1 = project_to_image(keypoints_3d,K2[i],tra2,rot2)

                    offset0 = calc_offset(keypoints_2d0).int()
                    offset1 = calc_offset(keypoints_2d1).int()
                
                    offset_prev = offset_prev | offset0
                    offset_gt = offset_gt | offset1

                    offset_delta = offset_pred[i]
                    offset_delta = torch.reshape(offset_pred[i].permute(1,2,0),image_shape+(9,2))
                    offset = offset_prev.cuda() + offset_delta
            
                    loss_offset = F.mse_loss(offset, offset_gt.float().cuda())

                mask_union = (mask1[i] & mask2[i])[:,:,0]
                mask_union = (mask_union != 0).float()
                conf_map = F.softmax(conf[i].flatten(), dim=0).reshape(image_shape)
                loss_mask = F.binary_cross_entropy(conf_map, mask_union)

                loss += loss_mask + loss_offset

            loss /= nb
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.cpu().detach().numpy())
            pbar.update(1)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'models/epoch='+str(epoch)+'.ckpt')


