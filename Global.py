import os.path as osp
import os

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

project_dir = os.path.dirname(os.path.abspath(__file__))

d = make_abs_path("./models")


ARCFACE_PRETRAINED_MODEL = osp.join(d, "resnet18_110.pth")