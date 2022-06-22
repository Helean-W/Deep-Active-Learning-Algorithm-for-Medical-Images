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
from albumentations import RandomRotate90, Resize
from dataset import Dataset
from utils import AverageMeter, str2bool
from collections import OrderedDict
import losses
from tqdm import tqdm
from metrics import iou_score, classify_accuracy
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
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
# NUM_ROUND = int((opts.nEnd - NUM_INIT_LB) / opts.nQuery)
NUM_ROUND = 18
DATA_NAME = 'ISIC2017'

## 得到所有图片和其对应类别
# X_tr, Y_tr, X_te, Y_te, train_img_ids, val_img_ids = get_dataset(DATA_NAME, opts.path)
X_tr, Y_tr, train_img_ids, val_img_ids = get_dataset(DATA_NAME, opts.path)

print("X_tr:", type(X_tr), '-', X_tr.shape)
print("Y_tr:", type(Y_tr), '-', Y_tr.shape)
# print("X_te:", type(X_te), '-', X_te.shape)
# print("Y_te:", type(Y_te), '-', Y_te.shape)
print("train_img_ids:", type(train_img_ids), '-', len(train_img_ids))
print("val_img_ids:", type(val_img_ids), '-', len(val_img_ids))

opts.dim = np.shape(X_tr)[1:]
print('dim:', opts.dim)
handler = get_handler('ISIC2017')


# start experiment
n_pool = len(train_img_ids)  #1600
n_test = len(val_img_ids)    #400
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
# idxs_lb = np.zeros(n_pool, dtype=bool)
# idxs_tmp = np.arange(n_pool)
# np.random.shuffle(idxs_tmp)
# idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = creat_balance_init_set(train_img_ids, NUM_INIT_LB)
idxs_lb[idxs_tmp] = True


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


#   初始标注样本id
now_train_img_ids = []
for i in (idxs_tmp[:NUM_INIT_LB]):
    now_train_img_ids.append(train_img_ids[i])
# now_train_img_ids = train_img_ids[idxs_tmp[:NUM_INIT_LB]]

init_train_dataset = Dataset(
    img_ids=now_train_img_ids,
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

init_train_loader = torch.utils.data.DataLoader(
    init_train_dataset,
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

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss_seg': AverageMeter(),
                  'loss_class': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, targetclass, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        targetclass = targetclass.cuda()
        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output, outclass, _ = model(input)
            loss_seg, loss_class = criterion(output, target, outclass, targetclass)
            iou,dice = iou_score(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss_seg.backward()
        loss_class.backward()
        optimizer.step()

        avg_meters['loss_seg'].update(loss_seg.item(), input.size(0))
        avg_meters['loss_class'].update(loss_class.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss_seg', avg_meters['loss_seg'].avg),
            ('loss_class', avg_meters['loss_class'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        data_tic = time.time()
    pbar.close()

    return OrderedDict([('loss_seg', avg_meters['loss_seg'].avg),
                        ('loss_class', avg_meters['loss_class'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss_seg': AverageMeter(),
                  'loss_class': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'classification': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, targetclass, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            targetclass = targetclass.cuda()
            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output, outclass, _ = model(input)
                loss_seg, loss_class = criterion(output, target, outclass, targetclass)
                iou,dice = iou_score(output, target)
                classify_auc = classify_accuracy(outclass, targetclass)

            avg_meters['loss_seg'].update(loss_seg.item(), input.size(0))
            avg_meters['loss_class'].update(loss_class.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['classification'].update(classify_auc, input.size(0))

            postfix = OrderedDict([
                ('loss_seg', avg_meters['loss_seg'].avg),
                ('loss_class', avg_meters['loss_class'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('classification', avg_meters['classification'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss_seg', avg_meters['loss_seg'].avg),
                        ('loss_class', avg_meters['loss_class'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('classification', avg_meters['classification'].avg)])


os.makedirs('models/UNext_BADGE', exist_ok=True)

criterion = losses.__dict__['BCEDiceLoss']().cuda()
cudnn.benchmark = True
net = archs.__dict__['UNext2'](1, 3, False)
net = net.cuda()

# params = filter(lambda p: p.requires_grad, net.parameters())
# optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)

# 对分类头设置更大学习率
classify_params = list(map(id, net.classHead.parameters()))
classify_params += list(map(id, net.classlinear.parameters()))

other_params = filter(lambda p: p.requires_grad and id(p) not in classify_params, net.parameters())

optimizer = optim.Adam([{'params': net.classHead.parameters(), 'lr': 0.0002, 'weight_decay': 0},
                        {'params': net.classlinear.parameters(), 'lr': 0.0002, 'weight_decay': 0},
                        {'params': other_params}
                        ], lr=0.0001, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

strategy = BadgeSamplingUnext(X_tr, Y_tr, idxs_lb, handler, val_transform)   #查询策略

log = OrderedDict([
    ('iteration', []),
    ('epoch', []),
    ('lr', []),
    ('loss_seg', []),
    ('loss_class', []),
    ('iou', []),
    ('val_loss_seg', []),
    ('val_loss_class', []),
    ('val_iou', []),
    ('val_dice', []),
    ('val_class_accu', []),
])

select_sample_ids = OrderedDict([
    ('iteration', []),
    ('id', [])])

best_iou = 0
trigger = 0

for epoch in range(1, 16):

    print('Epoch [%d/%d]' % (epoch, 15))

    # train for one epoch
    train_log = train(opts, init_train_loader, net, criterion, optimizer)
    # evaluate on validation set
    val_log = validate(opts, val_loader, net, criterion)

    scheduler.step()

    print('loss_seg %.4f - loss_class %.4f - iou %.4f - val_loss_seg %.4f - val_loss_class %.4f - val_iou %.4f - classify_accu %.4f'
          % (train_log['loss_seg'], train_log['loss_class'], train_log['iou'], val_log['loss_seg'], val_log['loss_class'], val_log['iou'], val_log['classification']))

    log['iteration'].append('0')
    log['epoch'].append(epoch)
    log['lr'].append(opts['lr'])
    log['loss_seg'].append(train_log['loss_seg'])
    log['loss_class'].append(train_log['loss_class'])
    log['iou'].append(train_log['iou'])
    log['val_loss_seg'].append(val_log['loss_seg'])
    log['val_loss_class'].append(val_log['loss_class'])
    log['val_iou'].append(val_log['iou'])
    log['val_dice'].append(val_log['dice'])
    log['val_class_accu'].append(val_log['classification'])

    pd.DataFrame(log).to_csv('models/UNext_BADGE/log.csv', index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(net.state_dict(), 'models/UNext_BADGE/model.pth')
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0

    # early stopping

    torch.cuda.empty_cache()



for rd in range(1, NUM_ROUND+1):

    new_best_iou = 0
    new_trigger = 0

    net.load_state_dict(torch.load('models/UNext_BADGE/model.pth'))

    output = strategy.query(NUM_QUERY, net)
    q_idxs = output
    idxs_lb[q_idxs] = True
    strategy.update(idxs_lb)
    # 更新标注集ID
    q_train_img_ids = []
    for i in q_idxs:
        q_train_img_ids.append(train_img_ids[i])

        select_sample_ids['id'].append(train_img_ids[i])
        select_sample_ids['iteration'].append(rd)
    now_train_img_ids = now_train_img_ids + q_train_img_ids

    pd.DataFrame(select_sample_ids).to_csv('models/UNext_BADGE/select_ids.csv', index=False)

    print('now number of train samples:', len(now_train_img_ids))
    train_dataset = Dataset(
        img_ids=now_train_img_ids,
        img_dir=os.path.join('data', opts['dataset'], 'images'),
        mask_dir=os.path.join('data', opts['dataset'], 'masks'),
        img_ext=opts['img_ext'],
        mask_ext=opts['mask_ext'],
        num_classes=opts['num_classes'],
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts['batch_size'],
        shuffle=True,
        num_workers=opts['num_workers'],
        drop_last=True)

    # 训练之前清空上一查询轮次后训练的权重
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
    net = net.apply(weight_reset).cuda()

    for epoch in range(1, 21):
        # train for one epoch
        train_log = train(opts, train_loader, net, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(opts, val_loader, net, criterion)

        scheduler.step()

        print('loss_seg %.4f - loss_class %.4f - iou %.4f - val_loss_seg %.4f - val_loss_class %.4f - val_iou %.4f - classify_accu %.4f'
              % (train_log['loss_seg'], train_log['loss_class'], train_log['iou'], val_log['loss_seg'], val_log['loss_class'], val_log['iou'], val_log['classification']))

        log['iteration'].append(rd)
        log['epoch'].append(epoch)
        log['lr'].append(opts['lr'])
        log['loss_seg'].append(train_log['loss_seg'])
        log['loss_class'].append(train_log['loss_class'])
        log['iou'].append(train_log['iou'])
        log['val_loss_seg'].append(val_log['loss_seg'])
        log['val_loss_class'].append(val_log['loss_class'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_class_accu'].append(val_log['classification'])

        pd.DataFrame(log).to_csv('models/UNext_BADGE/log.csv', index=False)

        new_trigger += 1

        if val_log['iou'] > new_best_iou:
            torch.save(net.state_dict(), 'models/UNext_BADGE/model.pth')
            new_best_iou = val_log['iou']
            print("=> saved best model")
            new_trigger = 0

        # early stopping

        torch.cuda.empty_cache()

