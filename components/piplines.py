import torch
import torch.nn as nn

from torchvision import models

from argparse import ArgumentParser


class ResNet_Baseline(nn.Module):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

    def __init__(self):
        super(ResNet_Baseline, self).__init__()
        self.backbone = models.resnet18()

    def forward(self, prev_frame, next_frame):
        stacked_image = torch.cat((prev_frame, next_frame), dim=1)

        out = self.backbone(stacked_image)

        return out


baseline = ResNet_Baseline()
prev_frame = torch.zeros(1,3,256,256)
next_frame = torch.zeros(1,3,256,256)
print(baseline(prev_frame,next_frame))
