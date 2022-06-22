import numpy as np
import sys
import gzip
import openml
import os
import argparse
from imblearn.over_sampling import SMOTE
import torchvision

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
from albumentations import RandomRotate90, Resize
from dataset import Dataset, DatasetClassify, ClassDataHandler
from utils import AverageMeter, str2bool
from collections import OrderedDict
import losses
from tqdm import tqdm
from metrics import iou_score, F1_Score, classification_result
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd

import time
from badge_sampling_unext import BadgeSamplingUnext
from balance_dataset import creat_balance_init_set

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=40)
parser.add_argument('--nStart', help='number of points to start', type=int, default=160)
parser.add_argument('--nEnd', help='total number of points to query', type=int, default=1600)
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

parser.add_argument('--num_workers', default=16, type=int)

opts = parser.parse_args()

# parameters
DATA_NAME = 'ISIC2017'

X_tr, Y_tr, train_img_ids, val_img_ids = get_dataset(DATA_NAME, opts.path)

# X_tr = np.zeros([20, 512])
# Y_tr = np.zeros([20, 1])
# Y_tr[:7] = 1

X_tr = X_tr.reshape((X_tr.shape[0], -1))
Y_tr = np.argmax(Y_tr, axis=1)

# resample
smo = SMOTE(n_jobs=-1)
x_sampling, y_sampling = smo.fit_resample(X_tr, Y_tr)
x_sampling = x_sampling.reshape(-1, 512, 512, 3)

print(x_sampling.shape, y_sampling.shape)

# img_ids = glob(os.path.join('data', 'ISIC2017', 'images', '*' + '.jpg'))
# img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]  ###得到文件名
# train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

print("train_img_ids:", type(train_img_ids), '-', len(train_img_ids))
print("val_img_ids:", type(val_img_ids), '-', len(val_img_ids))

# start experiment
n_pool = len(train_img_ids)  #1600
n_test = len(val_img_ids)    #400

print('number of testing pool: {}'.format(n_test), flush=True)


opts = vars(opts)
# dataloader
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

init_train_loader = torch.utils.data.DataLoader(ClassDataHandler(x_sampling, y_sampling, transform=train_transform),
                                                shuffle=True, batch_size=8, num_workers=16, drop_last=True)

# init_train_dataset = DatasetClassify(
#     img_ids=train_img_ids,
#     img_dir=os.path.join('data', opts['dataset'], 'images'),
#     img_ext=opts['img_ext'],
#     transform=train_transform)
val_dataset = Dataset(
    img_ids=val_img_ids,
    img_dir=os.path.join('data', opts['dataset'], 'images'),
    mask_dir=os.path.join('data', opts['dataset'], 'masks'),
    img_ext=opts['img_ext'],
    mask_ext=opts['mask_ext'],
    num_classes=opts['num_classes'],
    transform=val_transform)

# init_train_loader = torch.utils.data.DataLoader(
#     init_train_dataset,
#     batch_size=opts['batch_size'],
#     shuffle=True,
#     num_workers=opts['num_workers'],
#     drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opts['batch_size'],
    shuffle=False,
    num_workers=opts['num_workers'],
    drop_last=False)

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss_class': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, targetclass, _ in train_loader:
        input = input.cuda()
        targetclass = targetclass.cuda()
        # compute output
        outclass = model(input)
        loss_class = criterion(outclass, targetclass)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss_class.backward()
        optimizer.step()

        avg_meters['loss_class'].update(loss_class.item(), input.size(0))

        postfix = OrderedDict([
            ('loss_class', avg_meters['loss_class'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        data_tic = time.time()
    pbar.close()

    return OrderedDict([('loss_class', avg_meters['loss_class'].avg)])



def validate(val_loader, model, criterion,  all_pre, all_target):
    avg_meters = {'loss_class': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, targetclass, _ in val_loader:
            input = input.cuda()
            targetclass = targetclass.cuda()
            # compute output
            outclass = model(input)
            loss_class = criterion(outclass, targetclass)


            all_pre.append(outclass.cpu())
            all_target.append(targetclass.cpu())
            # classification_result(outclass, targetclass)

            avg_meters['loss_class'].update(loss_class.item(), input.size(0))

            postfix = OrderedDict([
                ('loss_class', avg_meters['loss_class'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    all_pre = np.concatenate(all_pre, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    classify_auc = F1_Score(all_pre, all_target)

    classification_result(all_pre, all_target)

    return OrderedDict([('loss_class', avg_meters['loss_class'].avg),
                        ('classification', classify_auc)])


os.makedirs('models/Resnet_test', exist_ok=True)

criterion_classify = losses.__dict__['ClassifyLoss']().cuda()
cudnn.benchmark = True
net = torchvision.models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 3)
net = net.cuda()

params = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(params, lr=0.0005, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)


log = OrderedDict([
    ('epoch', []),
    ('lr', []),
    ('loss_class', []),
    ('val_loss_class', []),
    ('val_class_accu', []),
])

best_classify = 0

for epoch in range(1, 100):
    all_pre = []
    all_target = []

    print('Epoch [%d/%d]' % (epoch, 15))

    # train for one epoch
    train_log = train(init_train_loader, net, criterion_classify, optimizer)
    # evaluate on validation set
    val_log = validate(val_loader, net, criterion_classify, all_pre, all_target)

    scheduler.step()

    print('loss_class %.4f - val_loss_class %.4f - classify_accu %.4f'
          % (train_log['loss_class'], val_log['loss_class'], val_log['classification']))

    log['epoch'].append(epoch)
    log['lr'].append(opts['lr'])
    log['loss_class'].append(train_log['loss_class'])
    log['val_loss_class'].append(val_log['loss_class'])
    log['val_class_accu'].append(val_log['classification'])

    pd.DataFrame(log).to_csv('models/Resnet_test/log.csv', index=False)

    if val_log['classification'] > best_classify:
        torch.save(net.state_dict(), 'models/Resnet_test/model.pth')
        best_classify = val_log['classification']
        print("=> saved best model")

    # early stopping

    torch.cuda.empty_cache()
