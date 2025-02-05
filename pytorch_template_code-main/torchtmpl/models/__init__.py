# coding: utf-8

# External imports
import torch
import torchvision.models.segmentation as torchmodels
from transformers import SegformerForSemanticSegmentation 

# Local imports
from .base_models import *
from .cnn_models import *
# from .UNet import UNet
from .UNet_improved import UNet

def build_model(cfg, input_size, num_classes):
    # Load DeepLabV3+ with ResNet backbone
    if cfg['class'] == "DeepLabV3":
        model = torchmodels.deeplabv3_resnet50(pretrained=False)
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        resnet_backbone = model.backbone

        # Modify the first convolutional layer to accept 1 channel
        resnet_backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=resnet_backbone.conv1.out_channels,
            kernel_size=resnet_backbone.conv1.kernel_size,
            stride=resnet_backbone.conv1.stride,
            padding=resnet_backbone.conv1.padding,
            bias=resnet_backbone.conv1.bias is not None
        )
        return model
    
    elif cfg['class'] == "SegFormer":
        # Load SegFormer model for semantic segmentation
        model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        
        # Modify the output classifier layer to match the required number of classes
        model.classifier = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

        # Modify the first convolutional layer to accept 1 channel for grayscale images
        model.backbone.embeddings.patch_embed.proj = torch.nn.Conv2d(
            in_channels=1,
            out_channels=model.backbone.embeddings.patch_embed.proj.out_channels,
            kernel_size=model.backbone.embeddings.patch_embed.proj.kernel_size,
            stride=model.backbone.embeddings.patch_embed.proj.stride,
            padding=model.backbone.embeddings.patch_embed.proj.padding,
            bias=model.backbone.embeddings.patch_embed.proj.bias is not None
        )
        
        return model

    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
