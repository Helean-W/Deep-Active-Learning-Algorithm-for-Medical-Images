import numpy as np
import sys
import gzip
import openml
import os
import argparse
from dataset import get_dataset, get_handler
from model import get_net
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import archs
import pdb
from scipy.stats import zscore

from glob import glob
from sklearn.model_selection import train_test_split
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90,Resize
from dataset import Dataset
from utils import AverageMeter, str2bool
from collections import OrderedDict
import losses
from tqdm import tqdm
from metrics import iou_score
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd

import time

from query_strategies import RandomSampling, BadgeSampling, \
    BaselineSampling, LeastConfidence, MarginSampling, \
    EntropySampling, CoreSet, ActiveLearningByLearning, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
    AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help='total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)

parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--deep_supervision', default=False, type=str2bool)
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
parser.add_argument('--input_w', default=256, type=int, help='image width')
parser.add_argument('--input_h', default=256, type=int, help='image height')
# dataset
parser.add_argument('--dataset', default='ISIC2017', help='dataset name')
parser.add_argument('--img_ext', default='.jpg', help='image file extension')
parser.add_argument('--mask_ext', default='.png', help='mask file extension')

parser.add_argument('--num_workers', default=24, type=int)

opts = parser.parse_args()


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output, _, _ = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        data_tic = time.time()
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output, _, _ = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


opts = vars(opts)

os.makedirs('models/UNext', exist_ok=True)

criterion = losses.__dict__['BCEDiceLoss']().cuda()
cudnn.benchmark = True
net = archs.__dict__['UNext'](1, 3, False)
net = net.cuda()
params = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)


# Data loading code
img_ids = glob(os.path.join('data', opts['dataset'], 'images', '*' + opts['img_ext']))
print(len(img_ids))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]  ###得到文件名

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

train_transform = Compose([
    RandomRotate90(),
    transforms.Flip(),
    Resize(opts['input_h'], opts['input_w']),
    transforms.Normalize(),
])

val_transform = Compose([
    Resize(opts['input_h'], opts['input_w']),
    transforms.Normalize(),
])

train_dataset = Dataset(
    img_ids=train_img_ids,
    img_dir=os.path.join('data', opts['dataset'], 'images'),
    mask_dir=os.path.join('data', opts['dataset'], 'masks'),
    img_ext=opts['img_ext'],
    mask_ext=opts['mask_ext'],
    num_classes=opts['num_classes'],
    transform=train_transform)
val_dataset = Dataset(
    img_ids=val_img_ids,
    img_dir=os.path.join('data', opts['dataset'], 'images'),
    mask_dir=os.path.join('data', opts['dataset'], 'masks'),
    img_ext=opts['img_ext'],
    mask_ext=opts['mask_ext'],
    num_classes=opts['num_classes'],
    transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opts['batch_size'],
    shuffle=True,
    num_workers=opts['num_workers'],
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opts['batch_size'],
    shuffle=False,
    num_workers=opts['num_workers'],
    drop_last=False)


log = OrderedDict([
    ('epoch', []),
    ('lr', []),
    ('loss', []),
    ('iou', []),
    ('val_loss', []),
    ('val_iou', []),
    ('val_dice', []),
])

best_iou = 0
trigger = 0

for epoch in range(500):
    print('Epoch [%d/%d]' % (epoch, 500))

    # train for one epoch
    train_log = train(opts, train_loader, net, criterion, optimizer)
    # evaluate on validation set
    val_log = validate(opts, val_loader, net, criterion)

    scheduler.step()

    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
          % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    log['epoch'].append(epoch)
    log['lr'].append(opts['lr'])
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])
    log['val_dice'].append(val_log['dice'])

    pd.DataFrame(log).to_csv('models/UNext/log.csv', index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(net.state_dict(), 'models/UNext/model.pth')
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0

    # early stopping

    torch.cuda.empty_cache()

##########################################################end
