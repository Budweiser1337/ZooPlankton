import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for the class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to logits to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Calculate the cross entropy loss
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        # Calculate p_t
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        # Calculate the Focal Loss components
        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Return per-element loss if no reduction
