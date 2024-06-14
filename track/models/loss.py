"""
loss funcs
"""

import torch 
import torch.nn as nn
import numpy as np 

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """ Focal loss class for edge classification

            focal_loss(probs) = -\alpha * (1 - probs) ^ \gamma * cross_etrophy(probs)
    
        Args:
            alpha, gamma: params, float
            num_classes: int
            size_average: bool, return mean of loss or sum
        
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  

            self.alpha = torch.Tensor(alpha)
        else:  # type(alpha) == float
            assert alpha < 1  

            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # [\alpha, 1 - \alpha, 1 - \alpha, ...]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        Args:
            preds: torch.Tensor, shape: (bs, num of samples, num of class)
            labels: torch.Tensor, shape: (bs, num of samples)

        Return:
            float

        """

        preds = preds.view(-1, preds.size(-1))  # shape: (bs * num_of_samples, num_of_class)
        self.alpha = self.alpha.to(preds.device)

        # cal log(softmax()) along 1-dim
        preds_logsoft = F.log_softmax(preds, dim=1)
        # cal exp(log(softmax())) = softmax, make 1-dim prob-like
        preds_softmax = torch.exp(preds_logsoft)

        # check the prob w.r.t. the true class
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # shape: (bs * num_of_samples, 1)
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))  # shape: (bs * num_of_samples, 1)
        alpha = self.alpha.gather(0, labels.view(-1))

        # - (1 - prob)^\gamma * \log{prob}
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        # dot with \alpha
        loss = torch.mul(alpha, loss.t())
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    

class FocalLossWithSigmoid(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="sum"):
        """ Focal loss class for edge classification, binary classification 
            with Sigmoid func

            focal_loss(probs) = -\alpha * (1 - probs) ^ \gamma * cross_etrophy(probs)
    
        Args:
            alpha, gamma: params, float
            num_classes: int
            reduction 'none' | 'mean' | 'sum'
                'none': No reduction will be applied to the output.
                'mean': The output will be averaged.
                'sum': The output will be summed.
        
        """

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, preds, labels):
        """
        Args:
            preds: torch.Tensor, shape: (num of samples, num of class) without sigmoided
            labels: torch.Tensor, shape: (num of samples)

        Return:
            float
        """

        preds = preds.float()
        labels = labels.float()
        probs = torch.sigmoid(preds)

        ce_loss = F.binary_cross_entropy_with_logits(
            input=preds, 
            target=labels, 
            reduction="none"
        )

        p_t = probs * labels + (1 - probs) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
    

class FocalLossAndDiceLoss(nn.Module):
    """ Focal loss (with sigmoid) and Dice loss

        loss = \beta_1 * focal_loss + \beta_2 * dice loss
    
    """

    def __init__(self, alpha=0.25, gamma=2, reduction="sum", smooth=1, beta1=0.8, beta2=0.2, 
                 return_weighted_sum=False) -> None:
        """
        Args:
            alpha, gamma: params, float
            num_classes: int
            reduction 'none' | 'mean' | 'sum'
                'none': No reduction will be applied to the output.
                'mean': The output will be averaged.
                'sum': The output will be summed.
            smooth: smooth coff used in dice loss to avoid divided by zero
            beta_1, beta_2: balance coff
            return_weighted_sum: bool, return focal loss and dice loss separately or weighted sum
        
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma 
        self.reduction = reduction
        self.smooth = smooth
        self.beta = [beta1, beta2]

        self.return_weighted_sum = return_weighted_sum

    def forward(self, preds, labels):
        """
        Args:
            preds: torch.Tensor, shape: (num of samples, num of class) without sigmoided
            labels: torch.Tensor, shape: (num of samples)

        Return:
            float, float: focal loss, dice loss
        """
        
        # cal focal loss
        preds = preds.float()
        labels = labels.float()
        probs = torch.sigmoid(preds)

        ce_loss = F.binary_cross_entropy_with_logits(
            input=preds, 
            target=labels, 
            reduction="none"
        )

        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()


        # cal dice loss
        pred_labels = (probs > 0.5).float()  # force prob w.r.t positive samples
        # larger than 0.5

        intersection = (pred_labels * labels).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_labels.sum() + labels.sum() + self.smooth)

        dice_loss = 1 - dice 

        if self.return_weighted_sum:
            return self.beta[0] * focal_loss + self.beta[1] * dice_loss
        else:
            return focal_loss, dice_loss


