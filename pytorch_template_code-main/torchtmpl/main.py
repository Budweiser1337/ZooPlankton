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
    # model.load_state_dict(torch.load("logs/UNet_best/best_model.pt")) 
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    # loss = optim.get_loss(config["loss"])
    if config["loss"]["name"] == "WeightedBCEWithLogitsLoss":
        pos_weight = torch.tensor(config["loss"]["params"]["pos_weight"], device="cuda" if torch.cuda.is_available() else "cpu")
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss = nn.BCEWithLogitsLoss()

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
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        # train_loss = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        # test_loss = utils.test(model, valid_loader, loss, device)

        # Train 1 epoch
        train_loss, train_metrics = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss, test_metrics = utils.test(model, valid_loader, loss, device)


        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
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
    model.load_state_dict(torch.load("model_logs/UNet_3/best_model.pt"))
    model.to(device)
    full_predictions = {}
    # Inference
    logging.info("= Running inference on the test set")
    predictions = []
    image_indices = []
    patch_positions = []

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(test_loader):
            images, row_starts, col_starts, img_indices = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Collect predictions

            for i in range(outputs.shape[0]):
                predictions.append(outputs[i].cpu().numpy())
                #predictions.append(((torch.sigmoid(outputs[i]).cpu().numpy()) >= 0.5).astype(np.int32))  # Move prediction to CPU
                patch_positions.append((row_starts[i].item(), col_starts[i].item()))
                image_indices.append(img_indices[i].item())

        logging.info("= Reconstructing full images from patches")
        reconstructed_images = {}
        for pred, (row_start, col_start), img_idx in zip(predictions, patch_positions, image_indices):
            if img_idx not in reconstructed_images:
                width, height = test_loader.dataset.image_sizes[img_idx]
                reconstructed_images[img_idx] = np.zeros((height, width), dtype=np.float32)

            patch_size = test_loader.dataset.patch_size
            row_end = row_start + patch_size
            col_end = col_start + patch_size
            reconstructed_images[img_idx][row_start:row_end, col_start:col_end] = pred.squeeze()
        binary_predictions = []
        for img_idx in sorted(reconstructed_images.keys()):
            binary_prediction = (reconstructed_images[img_idx] > 0.5).astype(np.uint8)  # Apply threshold
            binary_predictions.append(binary_prediction)

        print(f"Number of predictions: {len(binary_predictions)}")
        total_pixels = sum(pred.size for pred in binary_predictions)
        print(total_pixels)
        # Generate submission file
        # submission.generate_submission_file(binary_predictions, output_dir=config["prediction"]["dir"])

                
    # submission.generate_submission_file(full_predictions, output_dir=config["prediction"]["dir"])

    # for img_idx, prediction in full_predictions.items():
    #     save_path = os.path.join(config["prediction"]["dir"], f"prediction_{img_idx}.csv")
    #     pd.DataFrame(prediction).to_csv(save_path, index=False, header=False)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
