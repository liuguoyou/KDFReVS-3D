import torch.nn as nn
import torch
import cv2
from arcface.resnet import *
from torch.nn import functional as F
from collections import OrderedDict
import os.path as osp
from Global import ARCFACE_PRETRAINED_MODEL

class ArcFace():
    def __init__(self):
        self.model = resnet_face18(False)
        self.__load_state_dict__()
        self.model.eval()
        self.cri = nn.CosineEmbeddingLoss(margin=0.5)

    def __load_state_dict__(self):
        state_dictBA = torch.load(ARCFACE_PRETRAINED_MODEL)
        # create new OrderedDict that does not contain `module.`
        new_state_dictBA = OrderedDict()
        for k, v in state_dictBA.items():
            name = k[7:]  # remove `module.`
            new_state_dictBA[name] = v
        self.model.load_state_dict(new_state_dictBA)

    def to_gray(self, im):
        r = im[:, 0, :, :]
        g = im[:, 1, :, :]
        b = im[:, 2, :, :]

        im = 0.2989 * r + 0.5870 * g + 0.1140 * b
        im = im.unsqueeze(1)
        return im

    def arc_loss(self, im1, im2, targets):
        b, _, _, _ = im1.shape
        self.model.to(im1.device)
        im11 = F.interpolate(im1, (128, 128), mode="bilinear")
        im22 = F.interpolate(im2, (128, 128), mode="bilinear")

        im11 = self.to_gray(im11)
        im22 = self.to_gray(im22)

        im11_flip = im11[:, :, :, list(reversed(range(im11.shape[-1])))]
        im22_flip = im22[:, :, :, list(reversed(range(im22.shape[-1])))]

        im11 = torch.cat((im11, im11_flip))
        im22 = torch.cat((im22, im22_flip))

        out1 = self.model(im11)
        out2 = self.model(im22)

        fe_1_1 = out1[:b]
        fe_1_2 = out1[b:]
        fe_2_1 = out2[:b]
        fe_2_2 = out2[b:]

        f1 = torch.cat((fe_1_1, fe_1_2), dim=1)
        f2 = torch.cat((fe_2_1, fe_2_2), dim=1)

        f1 = F.normalize(f1)
        f2 = F.normalize(f2)

        cosin_loss = self.cri(f1, f2, targets)
        return cosin_loss



