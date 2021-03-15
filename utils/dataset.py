import os
import cv2
import glob

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import scipy.io as sio


class NOCS_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.folders = os.listdir(root_dir)

        self.num_scenes = len(self.folders)

        self.frame_list = []
        self.num_frames = []
        for folder in self.folders:
            self.frame_list.append(glob.glob(os.path.join(root_dir, folder, '*_color.png')))
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
            if num_frames + self.num_frames[i] - i > idx:
                idx1 = idx - num_frames
                idx2 = idx - num_frames + 1
                idx1 = self.frame_list[i][idx1][-14:-10]
                idx2 = self.frame_list[i][idx2][-14:-10]
                
                color1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_color.png'), cv2.IMREAD_COLOR)
                color2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_color.png'), cv2.IMREAD_COLOR)
                coord1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_coord.png'), cv2.IMREAD_COLOR)
                coord2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_coord.png'), cv2.IMREAD_COLOR)
                depth1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_depth.png'), cv2.IMREAD_COLOR)
                depth2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_depth.png'), cv2.IMREAD_COLOR)
                mask1 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx1+'_mask.png'), cv2.IMREAD_COLOR)
                mask2 = cv2.imread(os.path.join(self.root_dir, 'scene_'+str(i+1), idx2+'_mask.png'), cv2.IMREAD_COLOR)

                return {
                        'color1': color1,
                        'coord1': coord1,
                        'depth1': depth1,
                        'mask1': mask1,
                        'color2': color2,
                        'coord2': coord2,
                        'depth2': depth2,
                        'mask2': mask2
                        }
            
            num_frames += self.num_frames[i]


class YCB_Dataset(Dataset):
    def __init__(self, root_dir, length=3):
        self.root_dir = os.path.join(root_dir, 'data')
        self.folders = os.listdir(self.root_dir)

        self.num_scenes = len(self.folders)
        self.length = length

        self.num_frames = []
        for folder in self.folders:
            if folder.endswith('zip'):
                continue
            num_frame = len(os.listdir(os.path.join(self.root_dir, folder))) // 5
            self.num_frames.append(num_frame)

        self.models = {}
        for model_folder in os.listdir(os.path.join(root_dir, 'models')):
            self.models[model_folder[0:3]] = np.loadtxt(os.path.join(root_dir, 'models', model_folder, 'points.xyz'))

    def __len__(self):
        total_frames = 0
        for num_frame in self.num_frames:
            total_frames += num_frame - 1

        return total_frames

    def __getitem__(self, idx):
        num_frames = 0
        for i in range(self.num_scenes):
            if num_frames + self.num_frames[i] - i > idx:
                idx1 = idx - num_frames + 1
                idx2 = idx - num_frames + 2
                folder = os.path.join(self.root_dir, str(i).zfill(4))

                color1 = cv2.imread(os.path.join(folder, str(idx1).zfill(6)+'-color.png'), cv2.IMREAD_COLOR)
                color2 = cv2.imread(os.path.join(folder, str(idx2).zfill(6)+'-color.png'), cv2.IMREAD_COLOR)
                depth1 = cv2.imread(os.path.join(folder, str(idx1).zfill(6)+'-depth.png'), cv2.IMREAD_COLOR)
                depth2 = cv2.imread(os.path.join(folder, str(idx2).zfill(6)+'-depth.png'), cv2.IMREAD_COLOR)
                label1 = cv2.imread(os.path.join(folder, str(idx1).zfill(6)+'-label.png'), cv2.IMREAD_COLOR)
                label2 = cv2.imread(os.path.join(folder, str(idx2).zfill(6)+'-label.png'), cv2.IMREAD_COLOR)

                intrinsic1 = sio.loadmat(os.path.join(folder, str(idx1).zfill(6)+'-meta.mat'))['intrinsic_matrix']
                intrinsic2 = sio.loadmat(os.path.join(folder, str(idx2).zfill(6)+'-meta.mat'))['intrinsic_matrix']

                return {
                        'color1': color1,
                        'depth1': depth1,
                        'label1': label1,
                        'intrinsic1': intrinsic1,
                        'color2': color2,
                        'depth2': depth2,
                        'label2': label2,
                        'intrinsic2': intrinsic2
                        }

            num_frames += self.num_frames[i]

    def __getitem__(self, idx):
        pass


#dataset = NOCS_Dataset('/media/llewyn/TOSHIBA EXT/PoseEstimation/NOCS')
#print(dataset[0])
dataset = YCB_Dataset("/media/llewyn/TOSHIBA EXT/YCB_zip")
print(dataset.models)
