import numpy as np
import pdb
import cv2
import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from glob import glob
from sklearn.model_selection import train_test_split
import csv
import pandas as pd

import random
from balance_dataset import balance_dataset
from get_each_class_img_ids import get_each_class

def get_dataset(name, path):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'ISIC2017':
        return get_ISIC2017()

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))

    print('after 111111', data_tr.targets[1], np.array(data_tr.targets).shape)
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_ISIC2017():
    img_ids = glob(os.path.join('data', 'ISIC2017', 'images', '*' + '.jpg'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]  ###得到文件名

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=7)
    # 训练集重采样，测试集大小不变
    # train_img_ids = balance_dataset(train_img_ids)

    # img_class = csv.reader(open(os.path.join('data', 'ISIC2017', 'ISIC2017.csv')))
    img_class = pd.read_csv(os.path.join('data', 'ISIC2017', 'ISIC2017.csv'))

    train_class_data = []
    for id in train_img_ids:
        row = img_class[img_class['image_id'] == str(id)].values.tolist()
        row = row[0][1:]
        train_class_data.append(row)

    # val_class_data = []
    # for id in val_img_ids:
    #     row = img_class[img_class['image_id'] == str(id)].values.tolist()
    #     row = row[0][1:]
    #     val_class_data.append(row)

    train_class_data = torch.from_numpy(np.stack(train_class_data))
    # val_class_data = torch.from_numpy(np.stack(val_class_data))
    #print('**********', train_class_data.shape, val_class_data.shape)  #torch.Size([1600, 3]) torch.Size([400, 3])

    train_imgs = []
    for i in train_img_ids:
        img = cv2.imread(os.path.join('data', 'ISIC2017', 'images', i + '.jpg'))
        img = cv2.resize(img, (512, 512))
        train_imgs.append(img)

    # val_imgs = []
    # for i in val_img_ids:
    #     img = cv2.imread(os.path.join('data', 'ISIC2017', 'images', i + '.jpg'))
    #     img = cv2.resize(img, (512, 512))
    #     val_imgs.append(img)

    train_imgs = np.stack(train_imgs)
    # val_imgs = np.stack(val_imgs)
    #print('******************', val_imgs.shape, type(val_imgs))   #(400, 512, 512, 3) <class 'numpy.ndarray'>

    # return train_imgs, train_class_data, val_imgs, val_class_data, train_img_ids, val_img_ids
    return train_imgs, train_class_data, train_img_ids, val_img_ids

def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'ISIC2017':
        return DataHandlerISIC
    else:
        return DataHandler4

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        # type(y), y: 'torch.Tensor' tensor('num')
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandlerISIC(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            augmented = self.transform(image=x)
            img = augmented['image']
        img = img.astype('float32') / 255
        x = img.transpose(2, 0, 1)
        y = np.argmax(y)  # 将[0, 0, 1]转化成2
        return x, y, index

    def __len__(self):
        return len(self.X)


class Dataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.img_class = pd.read_csv(os.path.join('data', 'ISIC2017', 'ISIC2017.csv'))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                                                img_id + '_segmentation' + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        # get class
        row = self.img_class[self.img_class['image_id'] == img_id].values.tolist()
        row = row[0][1:]
        # 使用CE作为分类损失
        clas = row.index(1)  # 0 or 1 or 2

        # 使用focal loss作为分类损失
        # clas = np.array(row)
        return img, mask, clas, {'img_id': img_id}

# 欠采样
class DatasetClassify(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None):
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform
        self.img_class = pd.read_csv(os.path.join('data', 'ISIC2017', 'ISIC2017.csv'))
        self.img_ids = self.resample(img_ids)

    def resample(self, ids):
        num0, num1, num2 = 0, 0, 0
        class0, class1, class2 = [], [], []
        for id in ids:
            r = self.img_class[self.img_class['image_id'] == id].values.tolist()
            r = r[0][1:]
            c = r.index(1)
            if c == 0:
                num0 += 1
                class0.append(id)
            elif c == 1:
                num1 += 1
                class1.append(id)
            else:
                num2 += 1
                class2.append(id)

        t = (num0, num1, num2)
        m = min(t)

        if num0 > m:
            random.shuffle(class0)
            class0 = class0[:m]
        if num1 > m:
            random.shuffle(class1)
            class1 = class1[:m]
        if num2 > m:
            random.shuffle(class2)
            class2 = class2[:m]

        print(len(class0), len(class1), len(class2))
        res = class0 + class1 + class2
        random.shuffle(res)

        return res

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        # get class
        row = self.img_class[self.img_class['image_id'] == img_id].values.tolist()
        row = row[0][1:]
        # 使用CE作为分类损失
        clas = row.index(1)  # 0 or 1 or 2

        # 使用focal loss作为分类损失
        # clas = np.array(row)
        return img, clas, {'img_id': img_id}


# 过采样
class ClassDataHandler(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            augmented = self.transform(image=x)
            img = augmented['image']
        img = img.astype('float32') / 255
        x = img.transpose(2, 0, 1)
        # y = np.argmax(y)  # 将[0, 0, 1]转化成2
        return x, y, index

