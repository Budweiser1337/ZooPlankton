import torch

def precision(y_true, y_pred):
    numerator = (y_true*y_pred).sum()
    denominator = y_pred.sum()
    return numerator/denominator.clamp(min=1e-6)

def recall(y_true, y_pred):
    numerator = (y_true*y_pred).sum()
    denominator = y_true.sum()
    return numerator/denominator.clamp(min=1e-6)

def f1_score(y_true, y_pred):
    numerator = 2 * (y_true*y_pred).sum()
    denominator = (y_true.sum() + y_pred.sum())
    return numerator/denominator if denominator>0 else torch.tensor(0.0)

def compute_metrics(y_pred, y_true):
    return {
        "precision": precision(y_true=y_true, y_pred=y_pred).item(),
        "recall": recall(y_true=y_true, y_pred=y_pred).item(),
        "f1": f1_score(y_true=y_true, y_pred=y_pred).item()
    }

