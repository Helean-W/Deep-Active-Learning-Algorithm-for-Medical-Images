import os
import csv
import numpy as np
from glob import glob
import random


# img_ids = glob(os.path.join('data', 'ISIC2017', 'images', '*' + '.jpg'))
# img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]  ###得到文件名

def creat_balance_init_set(original_ids, num):
    class0, class1, class2 = [], [], []
    col_types = [str, int, int, int]
    with open(os.path.join('data', 'ISIC2017', 'ISIC2017.csv')) as f:
        f_csv = csv.reader(f)

        headers = next(f_csv)

        for row in f_csv:
            row = tuple(convert(value) for convert, value in zip(col_types, row))
            if row[0] in original_ids:
                if row[1] == 1:
                    class0.append(original_ids.index(row[0]))
                if row[2] == 1:
                    class1.append(original_ids.index(row[0]))
                elif row[3] == 1:
                    class2.append(original_ids.index(row[0]))

    count0 = len(class0)
    count1 = len(class1)
    count2 = len(class2)

    num_class0 = int(num / 3)
    num_class1 = num_class0
    num_class2 = num - num_class0 - num_class1

    if count0 < num_class0 or count1 < num_class1 or count2 < num_class2:
        print('类别无法平衡')
        if count0 < num_class0:
            num_class0 = count0
        if count1 < num_class1:
            num_class1 = count1
        num_class2 = num - num_class0 - num_class1

    class0 = np.array(class0)
    class1 = np.array(class1)
    class2 = np.array(class2)

    np.random.shuffle(class0)
    np.random.shuffle(class1)
    np.random.shuffle(class2)

    class0 = class0[:num_class0]
    class1 = class1[:num_class1]
    class2 = class2[:num_class2]

    balanced = np.concatenate((class0, class1, class2))

    return balanced


def balance_dataset(original_ids):
    balanced = []
    col_types = [str, int, int, int]

    class3 = []
    with open(os.path.join('data', 'ISIC2017', 'ISIC2017.csv')) as f:
        f_csv = csv.reader(f)

        headers = next(f_csv)

        for row in f_csv:
            row = tuple(convert(value) for convert, value in zip(col_types, row))
            if row[0] in original_ids:
                if row[1] == 1 or row[2] == 1:
                    balanced.append(row[0])
                elif row[3] == 1:
                    class3.append(row[0])

    sample_class3 = 880 - len(balanced)
    tmp = np.arange(len(class3))
    np.random.shuffle(tmp)

    for i in tmp[:sample_class3]:
        balanced.append(class3[i])

    return balanced

# if __name__ == '__main__':
#     creat_balance_init_set(img_ids)
