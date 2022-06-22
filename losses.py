import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'CrossEntropyFocalLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # #分类损失
        # classloss = F.cross_entropy(outClass, targetClass)

        # print('mask损失：', 0.5 * bce + dice, 'class损失:', classloss)
        # return 0.4 * (0.5 * bce + dice) + 0.6 * classloss
        return 0.5 * bce + dice


class ClassifyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outClass, targetClass):

        #分类损失
        classloss = F.cross_entropy(outClass, targetClass)

        return classloss


class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(CrossEntropyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):

        sigmoid_p = F.softmax(y_pred, 1)
        zeros = torch.zeros_like(sigmoid_p)
        pos_p_sub = torch.where(y_true > zeros, y_true - sigmoid_p, zeros)
        neg_p_sub = torch.where(y_true > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0))\
                              - (1 - self.alpha) * (neg_p_sub ** self.gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))

        batch_size = y_pred.shape[0]
        return per_entry_cross_ent.sum() / batch_size




class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


# if __name__ == '__main__':
#     y_pred = np.array([[-3, 1.5, 2.7], [-3, 1.5, 2.7], [-3, 1.5, 2.7]])
#     y_true = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
#
#     y_pred = torch.from_numpy(y_pred)
#     y_true = torch.from_numpy(y_true)
#
#     print(y_pred, y_true)
#     a = CrossEntropyFocalLoss()(y_pred, y_true)
#     print(a)
#     print(- 0.25 * (0.9974 ** 2) * log(0.0026) - (1 - 0.25) * (0.2309 ** 2) * log(1.0 - 0.2309) - (1 - 0.25) * (0.7666 ** 2) * log(1.0 - 0.7666))

