import numpy as np
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
from core_set_unext import CoreSet
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
NUM_INIT_LB = opts.nStart
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
# n_pool = 120
n_test = len(val_img_ids)    #400
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.seed(7)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# idxs_lb = np.zeros(n_pool, dtype=bool)
# idxs_tmp = creat_balance_init_set(train_img_ids, NUM_INIT_LB)
# idxs_lb[idxs_tmp] = True


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

# 初始分类集img array和class array

init_classify_imgs = X_tr[idxs_lb]
init_classify_labels = np.argmax(Y_tr, axis=1)[idxs_lb]

init_classify_imgs = init_classify_imgs.reshape((init_classify_imgs.shape[0], -1))

smo = SMOTE(n_jobs=-1)
x_sampling, y_sampling = smo.fit_resample(init_classify_imgs, init_classify_labels)
x_sampling = x_sampling.reshape(-1, 512, 512, 3)

init_train_dataset = Dataset(
    img_ids=now_train_img_ids,
    img_dir=os.path.join('data', opts['dataset'], 'images'),
    mask_dir=os.path.join('data', opts['dataset'], 'masks'),
    img_ext=opts['img_ext'],
    mask_ext=opts['mask_ext'],
    num_classes=opts['num_classes'],
    transform=train_transform)

# init_classify_dataset = DatasetClassify(
#     img_ids=now_train_img_ids,
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

init_train_loader = torch.utils.data.DataLoader(
    init_train_dataset,
    batch_size=opts['batch_size'],
    shuffle=True,
    num_workers=opts['num_workers'],
    drop_last=True)

# init_classify_loader = torch.utils.data.DataLoader(
#     init_classify_dataset,
#     batch_size=opts['batch_size'],
#     shuffle=True,
#     num_workers=opts['num_workers'],
#     drop_last=True)

init_classify_loader = torch.utils.data.DataLoader(ClassDataHandler(x_sampling, y_sampling, transform=train_transform),
                                                shuffle=True, batch_size=8, num_workers=16, drop_last=True)

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

def train_classify(train_loader, model, criterion, optimizer):
    avg_meters = {'loss_classify': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, targetclass, _ in train_loader:
        input = input.cuda()
        targetclass = targetclass.cuda()
        # compute output
        output, outclass, _ = model(input)
        loss = criterion(outclass, targetclass)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss_classify'].update(loss.item(), input.size(0))

        postfix = OrderedDict([
            ('loss_classify', avg_meters['loss_classify'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        data_tic = time.time()
    pbar.close()

    return OrderedDict([('loss_classify', avg_meters['loss_classify'].avg)])


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

            output, outclass, _ = model(input)
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


os.makedirs('models/UNext_Core_Set', exist_ok=True)

criterion_seg = losses.__dict__['BCEDiceLoss']().cuda()
criterion_classify = losses.__dict__['ClassifyLoss']().cuda()
cudnn.benchmark = True
net = archs.__dict__['UNext_core_set'](1, 3, False)
net = net.cuda()


classify_params = list(map(id, net.classHead.parameters()))
classify_params += list(map(id, net.classlinear.parameters()))
other_params = filter(lambda p: p.requires_grad and id(p) not in classify_params, net.parameters())
# params = filter(lambda p: p.requires_grad, net.parameters())

optimizer_seg = optim.Adam([{'params': other_params}], lr=0.0001, weight_decay=1e-4)
optimizer_classify = optim.Adam([{'params': net.classHead.parameters()},
                                {'params': net.classlinear.parameters()}],
                                lr=0.0002, weight_decay=1e-4)

# 对分类头设置更大学习率
# classify_params = list(map(id, net.classHead.parameters()))
# classify_params += list(map(id, net.classlinear.parameters()))
#
# other_params = filter(lambda p: p.requires_grad and id(p) not in classify_params, net.parameters())
#
# optimizer = optim.Adam([{'params': net.classHead.parameters(), 'lr': 0.0002, 'weight_decay': 0},
#                         {'params': net.classlinear.parameters(), 'lr': 0.0002, 'weight_decay': 0},
#                         {'params': other_params}
#                         ], lr=0.0001, weight_decay=1e-4)
scheduler_seg = lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=20, eta_min=1e-5)
scheduler_classify = lr_scheduler.CosineAnnealingLR(optimizer_classify, T_max=20, eta_min=1e-4)

strategy = CoreSet(X_tr, Y_tr, idxs_lb, handler, val_transform)   #查询策略

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
for epoch in range(1, 31):

    print('Segmentation Epoch [%d/%d]' % (epoch, 30))
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
        torch.save(net.state_dict(), 'models/UNext_Core_Set/model_seg.pth')
        best_iou = val_log['iou']
        print("=> saved best segmentation model")

    torch.cuda.empty_cache()

net.load_state_dict(torch.load('models/UNext_Core_Set/model_seg.pth'))


# 初始化分类
for epoch in range(1, 31):

    print('Classify Epoch [%d/%d]' % (epoch, 30))

    all_pre = []
    all_target = []
    train_log = train_classify(init_classify_loader, net, criterion_classify, optimizer_classify)
    val_log = validate(val_loader, net, criterion_seg, criterion_classify, all_pre, all_target)

    scheduler_classify.step()

    print('loss_classify %.4f - val_loss_seg %.4f - val_loss_classify %.4f - val_iou %.4f - classify_accu %.4f'
          % (train_log['loss_classify'], val_log['loss_seg'], val_log['loss_classify'], val_log['iou'], val_log['classification']))

    log['iteration'].append('0')
    log['stage'].append('classification')
    log['epoch'].append(epoch)
    log['lr'].append(opts['lr'])
    log['loss_seg'].append(' ')
    log['loss_classify'].append(train_log['loss_classify'])
    log['iou'].append(' ')
    log['val_loss_seg'].append(val_log['loss_seg'])
    log['val_loss_classify'].append(val_log['loss_classify'])
    log['val_iou'].append(val_log['iou'])
    log['val_dice'].append(val_log['dice'])
    log['val_class_accu'].append(val_log['classification'])

    if val_log['loss_classify'] <= best_classify:
        torch.save(net.state_dict(), 'models/UNext_Core_Set/model_classify.pth')
        best_classify = val_log['loss_classify']
        print("=> saved best classify model")

    torch.cuda.empty_cache()

pd.DataFrame(log).to_csv('models/UNext_Core_Set/log.csv', index=False)

for rd in range(1, NUM_ROUND+1):

    new_best_iou = 0
    new_best_classify = 10000

    net.load_state_dict(torch.load('models/UNext_Core_Set/model_classify.pth'))

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

    pd.DataFrame(select_sample_ids).to_csv('models/UNext_Core_Set/select_ids.csv', index=False)

    print('now itera is:', rd, 'now number of train samples:', len(now_train_img_ids))

    # resample
    now_classify_imgs = X_tr[idxs_lb]
    now_classify_labels = np.argmax(Y_tr, axis=1)[idxs_lb]

    now_classify_imgs = now_classify_imgs.reshape((now_classify_imgs.shape[0], -1))

    nowx_sampling, nowy_sampling = smo.fit_resample(now_classify_imgs, now_classify_labels)
    nowx_sampling = nowx_sampling.reshape(-1, 512, 512, 3)

    train_dataset = Dataset(
        img_ids=now_train_img_ids,
        img_dir=os.path.join('data', opts['dataset'], 'images'),
        mask_dir=os.path.join('data', opts['dataset'], 'masks'),
        img_ext=opts['img_ext'],
        mask_ext=opts['mask_ext'],
        num_classes=opts['num_classes'],
        transform=train_transform)

    # classify_dataset = DatasetClassify(
    #     img_ids=now_train_img_ids,
    #     img_dir=os.path.join('data', opts['dataset'], 'images'),
    #     img_ext=opts['img_ext'],
    #     transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts['batch_size'],
        shuffle=True,
        num_workers=opts['num_workers'],
        drop_last=True)

    # classify_loader = torch.utils.data.DataLoader(
    #     classify_dataset,
    #     batch_size=opts['batch_size'],
    #     shuffle=True,
    #     num_workers=opts['num_workers'],
    #     drop_last=True)

    classify_loader = torch.utils.data.DataLoader(
        ClassDataHandler(nowx_sampling, nowy_sampling, transform=train_transform),
        shuffle=True, batch_size=8, num_workers=16, drop_last=True)

    net = net.apply(weight_reset).cuda()


    c_params = list(map(id, net.classHead.parameters()))
    c_params += list(map(id, net.classlinear.parameters()))
    o_params = filter(lambda p: p.requires_grad and id(p) not in c_params, net.parameters())
    # params = filter(lambda p: p.requires_grad, net.parameters())

    optimizer_s = optim.Adam([{'params': o_params}], lr=0.0001, weight_decay=1e-4)
    optimizer_c = optim.Adam([{'params': net.classHead.parameters()}, {'params': net.classlinear.parameters()}],
                                    lr=0.0002, weight_decay=1e-4)

    scheduler_s = lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=30, eta_min=1e-5)
    scheduler_c = lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=30, eta_min=1e-4)


    # 训练分割网络
    for epoch in range(1, 31):
        print('segmentation Epoch [%d/%d]' % (epoch, 30))
        all_pre = []
        all_target = []
        train_log = train_seg(train_loader, net, criterion_seg, optimizer_s)
        val_log = validate(val_loader, net, criterion_seg, criterion_classify, all_pre, all_target)

        scheduler_s.step()

        print('loss_seg %.4f - iou %.4f - val_loss_seg %.4f - val_loss_classify %.4f - val_iou %.4f - classify_accu %.4f'
              % (train_log['loss_seg'], train_log['iou'], val_log['loss_seg'], val_log['loss_classify'], val_log['iou'], val_log['classification']))

        log['iteration'].append(rd)
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

        pd.DataFrame(log).to_csv('models/UNext_Core_Set/log.csv', index=False)

        if val_log['iou'] > new_best_iou:
            torch.save(net.state_dict(), 'models/UNext_Core_Set/model_seg.pth')
            new_best_iou = val_log['iou']
            print("=> saved best segmentation model")

        torch.cuda.empty_cache()

    net.load_state_dict(torch.load('models/UNext_Core_Set/model_seg.pth'))

    # 训练分类器
    for epoch in range(1, 31):
        print('Classify Epoch [%d/%d]' % (epoch, 30))
        all_pre = []
        all_target = []
        train_log = train_classify(classify_loader, net, criterion_classify, optimizer_c)
        val_log = validate(val_loader, net, criterion_seg, criterion_classify, all_pre, all_target)

        scheduler_c.step()

        print('loss_classify %.4f - val_loss_seg %.4f - val_loss_classify %.4f - val_iou %.4f - classify_accu %.4f'
              % (train_log['loss_classify'], val_log['loss_seg'], val_log['loss_classify'], val_log['iou'], val_log['classification']))

        log['iteration'].append(rd)
        log['stage'].append('classification')
        log['epoch'].append(epoch)
        log['lr'].append(opts['lr'])
        log['loss_seg'].append(' ')
        log['loss_classify'].append(train_log['loss_classify'])
        log['iou'].append(' ')
        log['val_loss_seg'].append(val_log['loss_seg'])
        log['val_loss_classify'].append(val_log['loss_classify'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_class_accu'].append(val_log['classification'])

        pd.DataFrame(log).to_csv('models/UNext_Core_Set/log.csv', index=False)

        if val_log['loss_classify'] <= new_best_classify:
            torch.save(net.state_dict(), 'models/UNext_Core_Set/model_classify.pth')
            new_best_classify = val_log['loss_classify']
            print("=> saved best classify model")

        torch.cuda.empty_cache()

