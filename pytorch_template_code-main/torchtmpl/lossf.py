import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for the class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):

        # Calculate the cross entropy loss
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        # Apply sigmoid to logits to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Calculate p_t
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = targets * self.alpha[1] + (1 - targets) * self.alpha[0]
            focal_weight = alpha_factor * focal_weight

        # Apply focal weight to loss
        loss = focal_weight * BCE_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Return per-element loss if no reduction

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to logits
        inputs = torch.sigmoid(inputs)

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice
    
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets):
        # Apply sigmoid to logits
        inputs = torch.sigmoid(inputs)

        # Compute gradients (edges) of predictions and targets
        pred_grad_x = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1])
        pred_grad_y = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :])

        target_grad_x = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
        target_grad_y = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])

        # Compute boundary loss (L1 distance between gradients)
        loss_x = torch.mean(torch.abs(pred_grad_x - target_grad_x))
        loss_y = torch.mean(torch.abs(pred_grad_y - target_grad_y))

        return (loss_x + loss_y) / 2  # Average over x and y directions
    
class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, inputs):
        # Apply sigmoid to logits
        inputs = torch.sigmoid(inputs)

        # Compute gradients (edges) of predictions
        grad_x = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1])
        grad_y = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :])

        # Compute smoothness loss (mean of gradients)
        return torch.mean(grad_x) + torch.mean(grad_y)

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, dice_weight, focal_weight, boundary_weight=0.1, smoothness_weight=0.1):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.boundary_loss = BoundaryLoss()
        self.smoothness_loss = SmoothnessLoss()

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        smoothness = self.smoothness_loss(inputs)
        return self.dice_weight * dice + self.focal_weight * focal + self.boundary_weight * boundary + self.smoothness_weight * smoothness
