import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


# def classify_accuracy(output, target):
#     count = 0
#     output = output.cpu().numpy()
#     batchsize = np.size(output, 0)
#     target = target.cpu().numpy()
#
#     output = np.argmax(output, axis=1)
#     for i, c in enumerate(output):
#         if c == target[i]:
#             count += 1
#     return count / batchsize


def F1_Score(output, target):
    # output = output.cpu().numpy()
    output = np.argmax(output, axis=1)
    # target = target.cpu().numpy()
    # target = np.argmax(target, axis=1)

    f1 = f1_score(target, output, average='micro')

    return f1

def classification_result(output, target):
    output = np.argmax(output, 1)

    # print(output, target)
    target_names = ['class0', 'class1', 'class2']
    result = classification_report(target, output, target_names=target_names)
    print(result)


