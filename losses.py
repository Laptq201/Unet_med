import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


__all__ = ["dice_coef", "dice_coef_necrotic", "dice_coef_edema", "dice_coef_enhancing",
           "dice_loss", "focal_loss", "total_loss", "accuracy", "precision", "sensitivity", "specificity",
           "dice_score", "Loss"]


def dice_coef(y_true, y_pred, epsilon=1e-6):
    """
    Dice coefficient
    """
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice


def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    """
    Dice coefficient for necrotic = core tumour
    """
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    return dice_coef(y_true, y_pred, epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    """
    Dice coefficient for edema = whole tumour
    """
    y_true = y_true[:, 2]
    y_pred = y_pred[:, 2]
    return dice_coef(y_true, y_pred, epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    """
    Dice coefficient for enhancing = active tumour
    """
    y_true = y_true[:, 3]
    y_pred = y_pred[:, 3]
    return dice_coef(y_true, y_pred, epsilon)


def dice_loss(y_true, y_pred):
    """
    Dice loss
    """
    smooth = 1e-5
    intersection = torch.sum(y_true * y_pred, axis=[1, 2, 3])
    union = torch.sum(y_true, axis=[1, 2, 3]) + \
        torch.sum(y_pred, axis=[1, 2, 3])
    dice = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice


def focal_loss(y_true, y_pred):
    """
    Focal loss = entropy loss for class imbalance
    """
    gamma = 2.0
    alpha = 0.25
    epsilon = 1e-5
    y_pred = torch.clip(y_pred, epsilon, 1.0-epsilon)
    cross_entropy = -y_true * torch.log(y_pred)
    focal = alpha * torch.pow(1-y_pred, gamma) * cross_entropy
    return torch.mean(focal, dim=0)


def total_loss(y_true, y_pred, w1=1, w2=2):
    """
    Total loss = sum of dice loss and focal loss
    """
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    total = w1*dice + w2*focal
    return total


"""
Metrics
"""


def accuracy(outputs, masks):
    correct = (outputs == masks).float().sum()
    total = outputs.numel()
    return correct/total


def precision(outputs, masks, class_id, epsilon=1e-7):
    true_positives = ((outputs == class_id) & (
        masks == class_id)).float().sum()
    false_positives = ((outputs == class_id) & (
        masks != class_id)).float().sum()
    return true_positives / (true_positives + false_positives + epsilon)


def sensitivity(outputs, masks, class_id, epsilon=1e-7):
    true_positives = ((outputs == class_id) & (
        masks == class_id)).float().sum()
    false_negatives = ((outputs != class_id) & (
        masks == class_id)).float().sum()
    return true_positives / (true_positives + false_negatives + epsilon)


def specificity(outputs, masks, class_id, epsilon=1e-7):
    # Same as precision
    true_negatives = ((outputs != class_id) & (
        masks != class_id)).float().sum()
    false_positives = ((outputs == class_id) & (
        masks != class_id)).float().sum()
    return true_negatives / (true_negatives + false_positives + epsilon)


def dice_score(outputs, masks, class_id, epsilon=1e-7):
    intersection = torch.sum((masks == class_id) * (outputs == class_id))
    union = torch.sum(masks == class_id) + torch.sum(outputs == class_id)
    return (2. * intersection + epsilon) / (union + epsilon)


"""

"""


class Loss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5,
                 num_classes=4, class_weights=None):
        super(Loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.num_classes = num_classes

        if class_weights is not None:
            self.num_classes = len(class_weights)
            device = torch.device(f"cuda")
            class_weights = torch.tensor(
                class_weights, dtype=torch.float32).to(device)

        self.ce_loss = CrossEntropyLoss(weight=class_weights)
        print(
            f"num_classes: {self.num_classes}, class_weights: {class_weights}")

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = F.softmax(pred, dim=1)
        total_loss = 0
        for i in range(self.num_classes):
            pred_i = pred[:, i, ...]
            target_i = (target == i).float()

            intersection = torch.sum(pred_i * target_i)
            union = torch.sum(pred_i+target_i)
            dice = (2. * intersection + smooth) / (union + smooth)
            total_loss += (1 - dice)
        return total_loss / self.num_classes

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(torch.clamp(logits, min=-100, max=100), targets)

        dice = self.dice_loss(logits, targets)

        return self.weight_ce * ce_loss + self.weight_dice * dice
