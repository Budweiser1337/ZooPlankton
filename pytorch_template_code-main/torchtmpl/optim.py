# coding: utf-8

# External imports
import torch
import torch.nn as nn

# Local imports
import lossf


def get_loss(lossname, config):
    if lossname == "WeightedBCEWithLogitsLoss":
        pos_weight = torch.tensor(config["loss"]["params"]["pos_weight"], device="cuda" if torch.cuda.is_available() else "cpu")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif lossname == "FocalLoss":
        alpha = config["loss"]["params"]["alpha"]
        gamma = config["loss"]["params"]["gamma"]
        return lossf.FocalLoss(alpha=alpha, gamma=gamma)
    return eval(f"nn.{config['loss']}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
