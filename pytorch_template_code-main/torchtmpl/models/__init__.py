# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
# from .UNet import UNet
from .UNet_improved import UNet

def build_model(cfg, input_size, num_classes):
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
