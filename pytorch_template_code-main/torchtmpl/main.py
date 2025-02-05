# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torch.nn as nn
import torchinfo.torchinfo as torchinfo
import numpy as np
from tqdm import tqdm
import torchvision.models.segmentation as torchmodels


# Local imports
import data
import models
import optim
import utils
import submission

def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if use_cuda:
        print("using gpu")
    else:
        print("using cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size[0], 1)
    # model.load_state_dict(torch.load("/usr/users/sdim/sdim_22/team-6-kaggle-challenge-deep-learning/pytorch_template_code-main/model_logs/UNet_8/best_model.pt")) 
    # Load DeepLabV3+ with ResNet backbone
    # model = torchmodels.deeplabv3_resnet50(pretrained=False)
    # num_classes = 1
    # model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    # resnet_backbone = model.backbone

    # # Modify the first convolutional layer to accept 1 channel
    # resnet_backbone.conv1 = torch.nn.Conv2d(
    #     in_channels=1,
    #     out_channels=resnet_backbone.conv1.out_channels,
    #     kernel_size=resnet_backbone.conv1.kernel_size,
    #     stride=resnet_backbone.conv1.stride,
    #     padding=resnet_backbone.conv1.padding,
    #     bias=resnet_backbone.conv1.bias is not None
    # )
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"], config)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss, train_metrics = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss, test_metrics = utils.test(model, valid_loader, loss, device)


        updated = model_checkpoint.update(test_metrics["f1"])
        logging.info(
            "[%d/%d] Test F1-score : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_metrics["f1"],
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss,
                   **{"train_"+k:v for k,v in train_metrics.items()},
                   **{"test_"+k:v for k,v in test_metrics.items()}}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)

def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if use_cuda:
        print("using gpu")
    else:
        print("using cpu")

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    test_loader, input_size, num_classes = data.get_test_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, 1, 1)
    model.load_state_dict(torch.load("model_logs/UNet_12/model.pt"))
    model.to(device)

    # Inference
    logging.info("= Running inference on the test set")
    predictions = []
    image_indices = []
    patch_positions = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, row_starts, col_starts, img_indices = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            outputs = (torch.sigmoid(outputs) > .5).byte().cpu().numpy()
            # Collect predictions
            for i in range(outputs.shape[0]):
                predictions.append(outputs[i])
                patch_positions.append((row_starts[i].item(), col_starts[i].item()))
                image_indices.append(img_indices[i].item())

        logging.info("= Reconstructing full images from patches")
        reconstructed_images = {}
        for pred, (row_start, col_start), img_idx in zip(predictions, patch_positions, image_indices):
            width, height = test_loader.dataset.image_sizes[img_idx]
            if img_idx not in reconstructed_images:
                reconstructed_images[img_idx] = np.zeros((height, width), dtype=np.float32)

            patch_size = test_loader.dataset.patch_size
            row_end = min(row_start + patch_size, height)
            col_end = min(col_start + patch_size, width)

            valid_patch_height = row_end - row_start
            valid_patch_width = col_end - col_start
            
            reconstructed_images[img_idx][row_start:row_end, col_start:col_end] = pred[0, :valid_patch_height, :valid_patch_width]

        submission.generate_submission_file(list(reconstructed_images.values()), output_dir=config["prediction"]["dir"])

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
