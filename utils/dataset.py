import os
import cv2

import torch
from torch.utils.data import DataLoader, Dataset

class NOCS_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.folders = os.listdir(root_dir)

        self.num_scenes = len(self.folders)

        self.num_frames = []
        for folder in self.folders:
            num_frame = (len(os.listdir(os.path.join(root_dir, folder))) - 3) // 5
            self.num_frames.append(num_frame)

    def __len__(self):
        total_frames = 0
        for num_frame in self.num_frames:
            total_frames += num_frame - 1

        return total_frames

    def __getitem__(self, idx):
        num_frames = 0
        for i in range(self.num_scenes):
            if num_frames + self.num_frames[i] - 1 > idx:
                idx1 = str(idx-num_frames).zfill(4)
                idx2 = str(idx-num_frames+1).zfill(4)
                
                color1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_color.png'), cv2.IMREAD_COLOR)
                color2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_color.png'), cv2.IMREAD_COLOR)
                coord1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_coord.png'), cv2.IMREAD_COLOR)
                coord2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_coord.png'), cv2.IMREAD_COLOR)
                depth1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_depth.png'), cv2.IMREAD_COLOR)
                depth2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_depth.png'), cv2.IMREAD_COLOR)
                mask1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_mask.png'), cv2.IMREAD_COLOR)
                mask2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_mask.png'), cv2.IMREAD_COLOR)

                return color1, coord1, depth1, mask1, color2, coord2, depth2, mask2
            
            num_frames += self.num_frames[i] - 1


class YCB_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

#dataset = NOCS_Dataset('/media/llewyn/TOSHIBA EXT/PoseEstimation/NOCS')
#print(dataset[0])

