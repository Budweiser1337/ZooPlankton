# coding: utf-8

# External imports
import torch
import torchvision.models.segmentation as torchmodels
import segmentation_models_pytorch as smp
from monai.networks.nets import SwinUNETR

# Local imports
from .base_models import *
from .cnn_models import *
from .UNet import UNet
# from .UNet_improved import UNet

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
    
    elif cfg['class'] == "Segformer":
        # Load SegFormer model for semantic segmentation
        model = smp.Segformer(
            encoder_name="timm-efficientnet-b3",
            encoder_weights="imagenet",
            classes=1,
            activation=None, # Use raw logits (for BCE/Focal/Dice loss)
            in_channels=1,
        )
        return model
    
    elif cfg['class'] == 'UnetPlus':
        model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b6",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None,  # Use raw logits (for BCE/Focal/Dice loss)
            decoder_attention_type="scse"
        )
        return model
    
    elif cfg['class'] == "SwinUNETR":
        model = SwinUNETR(
            img_size=(512, 512),
            in_channels=1,
            out_channels=1,
            feature_size=48,  # Controls model complexity
            spatial_dims=2
        )
        return model
    
    elif cfg['class'] == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None
        )
        return model
    
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
