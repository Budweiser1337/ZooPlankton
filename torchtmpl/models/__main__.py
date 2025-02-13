# coding: utf-8

# External imports
import torch

# Local imports
from . import build_model, UNet


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")

def test_unet():
    model = UNet(3, 1)
    model.to('cuda')
    input_size = (3, 512, 512)
    batch_size = 8
    input_tensor = torch.randn(batch_size, *input_size).to('cuda')
    output = model(input_tensor)
    print(output.shape)

if __name__ == "__main__":
    test_unet()
