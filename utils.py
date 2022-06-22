import argparse
import torch.nn as nn
import os
import csv

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def csvAddCol():
    with open(os.path.join('data', 'ISIC2017', 'ISIC-2017_Training_Part3_GroundTruth.csv')) as csvFile:
        rows = csv.reader(csvFile)
        with open(os.path.join('data', 'ISIC2017', 'ISIC2017.csv'), 'w') as f:
            writer = csv.writer(f)
            for row in rows:
                if row[1] == '0.0' and row[2] == '0.0':
                    row.append('1')
                    writer.writerow(row)
                else:
                    row.append('0')
                    writer.writerow(row)

