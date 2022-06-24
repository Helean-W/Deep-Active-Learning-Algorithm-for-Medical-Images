import numpy as np
import os
import argparse

from torch import nn
import torch
import archs

from glob import glob
from sklearn.model_selection import train_test_split
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90, Resize
from dataset import Dataset, DatasetClassify, get_dataset, get_handler, ClassDataHandler
from utils import AverageMeter, str2bool
from collections import OrderedDict
import losses
from tqdm import tqdm
from metrics import iou_score, F1_Score
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd

import time
from badge_sampling_unext import BadgeSamplingUnext
from balance_dataset import creat_balance_init_set
from imblearn.over_sampling import SMOTE

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=32)
parser.add_argument('--nStart', help='number of points to start', type=int, default=80)
parser.add_argument('--nEnd', help='total number of points to query', type=int, default=1600)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)

parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--deep_supervision', default=False, type=str2bool)
parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
parser.add_argument('--input_w', default=512, type=int, help='image width')
parser.add_argument('--input_h', default=512, type=int, help='image height')
# dataset
parser.add_argument('--dataset', default='ISIC2017', help='dataset name')
parser.add_argument('--img_ext', default='.jpg', help='image file extension')
parser.add_argument('--mask_ext', default='.png', help='mask file extension')

parser.add_argument('--num_workers', default=16, type=int)

opts = parser.parse_args()

# parameters
NUM_INIT_LB = opts.nStart + 64
NUM_QUERY = opts.nQuery
# NUM_ROUND = int((opts.nEnd - NUM_INIT_LB) / opts.nQuery)
NUM_ROUND = 16
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
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
# np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True


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


# #   初始标注样本id
# now_train_img_ids = []
# for i in (idxs_tmp[:NUM_INIT_LB]):
#     now_train_img_ids.append(train_img_ids[i])
# # now_train_img_ids = train_img_ids[idxs_tmp[:NUM_INIT_LB]]

now_train_img_ids = train_img_ids[:NUM_INIT_LB]

print(now_train_img_ids)

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
    shuffle=False,
    num_workers=opts['num_workers'],
    drop_last=True)


val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opts['batch_size'],
    shuffle=False,
    num_workers=opts['num_workers'],
    drop_last=False)

def train_seg(train_loader, model, criterion, optimizer):
    avg_meters = {'loss_seg': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        # compute output
        output, outclass, _ = model(input)
        loss = criterion(output, target)
        iou, dice = iou_score(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss_seg'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss_seg', avg_meters['loss_seg'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        data_tic = time.time()
    pbar.close()

    return OrderedDict([('loss_seg', avg_meters['loss_seg'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(val_loader, model, criterion_seg, criterion_classify, all_pre, all_target):
    avg_meters = {'loss_seg': AverageMeter(),
                  'loss_classify': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, targetclass, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            targetclass = targetclass.cuda()

            # tmp = np.ones((8, 1, 512, 512))
            # tmp = torch.Tensor(tmp)

            output, outclass, _ = model(input)
            # print(type(output), output.shape)
            loss_seg = criterion_seg(output, target)
            loss_class = criterion_classify(outclass, targetclass)
            iou,dice = iou_score(output, target)
            # classify_f1 = F1_Score(outclass, targetclass)

            all_pre.append(outclass.cpu())
            all_target.append(targetclass.cpu())

            avg_meters['loss_seg'].update(loss_seg.item(), input.size(0))
            avg_meters['loss_classify'].update(loss_class.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss_seg', avg_meters['loss_seg'].avg),
                ('loss_classify', avg_meters['loss_classify'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    all_pre = np.concatenate(all_pre, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    classify_f1 = F1_Score(all_pre, all_target)

    return OrderedDict([('loss_seg', avg_meters['loss_seg'].avg),
                        ('loss_classify', avg_meters['loss_classify'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('classification', classify_f1)])


os.makedirs('models/UNext_BADGE_Seg_Class', exist_ok=True)

criterion_seg = losses.__dict__['BCEDiceLoss']().cuda()
criterion_classify = losses.__dict__['ClassifyLoss']().cuda()
cudnn.benchmark = True
net = archs.__dict__['UNext2'](1, 3, False)
net = net.cuda()


classify_params = list(map(id, net.classHead.parameters()))
classify_params += list(map(id, net.classlinear.parameters()))
other_params = filter(lambda p: p.requires_grad and id(p) not in classify_params, net.parameters())
# params = filter(lambda p: p.requires_grad, net.parameters())

optimizer_seg = optim.Adam([{'params': other_params}], lr=0.0001, weight_decay=1e-4)


scheduler_seg = lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=20, eta_min=1e-5)

# strategy = BadgeSamplingUnext(X_tr, Y_tr, idxs_lb, handler, val_transform)   #查询策略

log = OrderedDict([
    ('iteration', []),
    ('stage', []),
    ('epoch', []),
    ('lr', []),
    ('loss_seg', []),
    ('loss_classify', []),
    ('iou', []),
    ('val_loss_seg', []),
    ('val_loss_classify', []),
    ('val_iou', []),
    ('val_dice', []),
    ('val_class_accu', []),
])

select_sample_ids = OrderedDict([
    ('iteration', []),
    ('id', [])])

best_iou = 0
best_classify = 10000 #大值

# 训练之前清空上一查询轮次后训练的权重
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

net = net.apply(weight_reset).cuda()

# 初始化分割
for epoch in range(1, 16):

    print('Segmentation Epoch [%d/%d]' % (epoch, 20))
    all_pre = []
    all_target = []

    train_log = train_seg(init_train_loader, net, criterion_seg, optimizer_seg)
    val_log = validate(val_loader, net, criterion_seg, criterion_classify, all_pre, all_target)


    scheduler_seg.step()

    print('loss_seg %.4f  - iou %.4f - val_loss_seg %.4f - val_loss_class %.4f - val_iou %.4f - classify_accu %.4f'
          % (train_log['loss_seg'], train_log['iou'], val_log['loss_seg'], val_log['loss_classify'], val_log['iou'], val_log['classification']))

    log['iteration'].append('0')
    log['stage'].append('segmentation')
    log['epoch'].append(epoch)
    log['lr'].append(opts['lr'])
    log['loss_seg'].append(train_log['loss_seg'])
    log['loss_classify'].append(' ')
    log['iou'].append(train_log['iou'])
    log['val_loss_seg'].append(val_log['loss_seg'])
    log['val_loss_classify'].append(val_log['loss_classify'])
    log['val_iou'].append(val_log['iou'])
    log['val_dice'].append(val_log['dice'])
    log['val_class_accu'].append(val_log['classification'])

    if val_log['iou'] > best_iou:
        torch.save(net.state_dict(), 'models/UNext_BADGE_Seg_Class/model_seg.pth')
        best_iou = val_log['iou']
        print("=> saved best segmentation model")

    torch.cuda.empty_cache()

# net.load_state_dict(torch.load('models/UNext_BADGE_Seg_Class/model_seg.pth'))
#
# NUM_INIT_LB += 32
#
# now_train_img_ids = train_img_ids[:NUM_INIT_LB]
#
# print(len(now_train_img_ids))
# train_dataset = Dataset(
#     img_ids=now_train_img_ids,
#     img_dir=os.path.join('data', opts['dataset'], 'images'),
#     mask_dir=os.path.join('data', opts['dataset'], 'masks'),
#     img_ext=opts['img_ext'],
#     mask_ext=opts['mask_ext'],
#     num_classes=opts['num_classes'],
#     transform=train_transform)
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=opts['batch_size'],
#     shuffle=True,
#     num_workers=opts['num_workers'],
#     drop_last=True)
#
#
# net = net.apply(weight_reset).cuda()
#
# c_params = list(map(id, net.classHead.parameters()))
# c_params += list(map(id, net.classlinear.parameters()))
# o_params = filter(lambda p: p.requires_grad and id(p) not in c_params, net.parameters())
# optimizer_2 = optim.Adam(o_params, lr=0.0001, weight_decay=1e-4)
#
# scheduler_2 = lr_scheduler.CosineAnnealingLR(optimizer_2, 20, 1e-5)
#
# for epoch in range(1, 16):
#
#     print('Segmentation Epoch [%d/%d]' % (epoch, 20))
#     all_pre = []
#     all_target = []
#
#     train_log = train_seg(train_loader, net, criterion_seg, optimizer_2)
#     val_log = validate(val_loader, net, criterion_seg, criterion_classify, all_pre, all_target)
#
#     scheduler_2.step()
#
#     print('loss_seg %.4f  - iou %.4f - val_loss_seg %.4f - val_loss_class %.4f - val_iou %.4f - classify_accu %.4f'
#           % (train_log['loss_seg'], train_log['iou'], val_log['loss_seg'], val_log['loss_classify'], val_log['iou'], val_log['classification']))
#
#     log['iteration'].append('0')
#     log['stage'].append('segmentation')
#     log['epoch'].append(epoch)
#     log['lr'].append(opts['lr'])
#     log['loss_seg'].append(train_log['loss_seg'])
#     log['loss_classify'].append(' ')
#     log['iou'].append(train_log['iou'])
#     log['val_loss_seg'].append(val_log['loss_seg'])
#     log['val_loss_classify'].append(val_log['loss_classify'])
#     log['val_iou'].append(val_log['iou'])
#     log['val_dice'].append(val_log['dice'])
#     log['val_class_accu'].append(val_log['classification'])
#
#     if val_log['iou'] > best_iou:
#         torch.save(net.state_dict(), 'models/UNext_BADGE_Seg_Class/model_seg.pth')
#         best_iou = val_log['iou']
#         print("=> saved best segmentation model")
#
#     torch.cuda.empty_cache()