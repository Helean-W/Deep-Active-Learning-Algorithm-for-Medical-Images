import os
import csv


def get_each_class():
    class0, class1, class2 = [], [], []
    col_types = [str, int, int, int]

    with open(os.path.join('data', 'ISIC2017', 'ISIC2017.csv')) as f:
        f_csv = csv.reader(f)

        for row in f_csv:
            row = tuple(convert(value) for convert, value in zip(col_types, row))
            if row[1] == 1:
                class0.append(row[0])
            elif row[2] == 1:
                class1.append(row[0])
            elif row[3] == 1:
                class2.append(row[0])

    return class0, class1, class2

