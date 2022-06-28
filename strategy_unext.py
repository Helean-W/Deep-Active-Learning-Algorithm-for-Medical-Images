import heapq

import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
import resnet


class Strategy_Unext:
    def __init__(self, X, Y, idxs_lb, handler, val_transform):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.handler = handler
        self.val_transform = val_transform
        self.n_pool = len(Y)

    def query(self, n, net):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model):
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = 3   # 3类
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.val_transform),
                               shuffle=False, batch_size=8, num_workers=16)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                # print('查询loader:', x.shape, y.shape, idxs)  torch.Size([16, 3, 512, 512]) torch.Size([16])  torch.Size([16])
                x, y = Variable(x.cuda()), Variable(y.cuda())
                _, cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        -1 * batchProbs[j][c])

            return torch.Tensor(embedding)


class Strategy_Seg_BADGE:
    def __init__(self, X, Y, idxs_lb, handler, val_transform):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.handler = handler
        self.val_transform = val_transform
        self.n_pool = len(Y)

    def query(self, n, net):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model):
        embDim = model.get_seg_channsels()
        model.eval()
        K = 5   # 每张图的激活值分成5组
        embedding = np.zeros([len(Y), embDim * K])
        loader_te = DataLoader(self.handler(X, Y, transform=self.val_transform),
                               shuffle=False, batch_size=8, num_workers=16)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                # print('查询loader:', x.shape, y.shape, idxs)  torch.Size([8, 3, 512, 512]) torch.Size([8])  torch.Size([8])
                x, y = Variable(x.cuda()), Variable(y.cuda())
                mask, emb, _, _ = model(x)  # mask: torch.Size([8, 1, 512, 512])  emb: torch.Size([8, 16, 512, 512])
                emb = emb.cpu().numpy()
                batchProbs = torch.sigmoid(mask).cpu().numpy()
                batchProbs = np.squeeze(batchProbs)  # torch.Size([8, 512, 512])

                final_mask = batchProbs.copy()  # torch.Size([8, 512, 512])
                final_mask[final_mask >= 0.5] = 1
                final_mask[final_mask < 0.5] = 0


                emb = torch.randn(8, 16, 512, 512)
                mask = torch.randn(8, 1, 512, 512)
                batchProbs = torch.sigmoid(mask).repeat(1, 16, 1, 1)
                bins_embedding = []
                for base in range(0, 10, 2):
                    bin_embedding = deepcopy(emb)
                    bin_embedding[batchProbs < base / 10] = 0
                    bin_embedding[batchProbs > (base + 2) / 10] = 0
                    bin_embedding = torch.sum(bin_embedding, (-1, -2)) / torch.sum(bin_embedding[:, :1] != 0,
                                                                                   dim=(-1, -2))
                    bins_embedding.append(bin_embedding)
                bins_embedding = torch.cat(bins_embedding, dim=1)
                print(bins_embedding.shape)


                bce_grad_res = np.zeros([8, 512, 512, embDim])   # sigmoid结果的梯度嵌入结果

                # for j in range(len(y)):
                #     singleProbs = batchProbs[j]
                #     singleProbs = singleProbs.flatten() - 0.5
                #     singleProbs = np.abs(singleProbs)
                #
                #     index = heapq.nsmallest(50, range(len(singleProbs)), singleProbs.take)
                #     index = np.array(index)
                #     r, c = divmod(index, 512)
                #
                #     for i in range(len(r)):
                #
                #         # print('batchprobs:', batchProbs[j][r[i]][c[i]], final_mask[j][r[i]][c[i]])
                #         if final_mask[j][r[i]][c[i]] == 1:
                #             embedding[idxs[j]][embDim * i: embDim * (i + 1)] = deepcopy(emb[j, :, r[i], c[i]]) * (1 - batchProbs[j][r[i]][c[i]])
                #         else:
                #             embedding[idxs[j]][embDim * i: embDim * (i + 1)] = deepcopy(emb[j, :, r[i], c[i]]) * (-1 * batchProbs[j][r[i]][c[i]])

                # for j in range(len(y)):
                #     range02, range24, range46, range68, range80 = np.zeros([16]), np.zeros([16]), np.zeros([16]), np.zeros([16]), np.zeros([16])
                #     count02, count24, count46, count68, count80 = 0, 0, 0, 0, 0
                #     for row in range(512):
                #         for col in range(512):
                #             if final_mask[j][row][col] == 1:
                #                 bce_grad_res[j][row][col] = deepcopy(emb[j, :, row, col]) * (1 - batchProbs[j][row][col])  # emb: torch.Size([8, 16, 512, 512]),取出第1维的16个数
                #             else:
                #                 bce_grad_res[j][row][col] = deepcopy(emb[j, :, row, col]) * (-1 * batchProbs[j][row][col])
                #
                #             if 0.2 >= batchProbs[j][row][col] >= 0.0:
                #                 range02 += bce_grad_res[j][row][col]
                #                 count02 += 1
                #             elif 0.4 >= batchProbs[j][row][col] > 0.2:
                #                 range24 += bce_grad_res[j][row][col]
                #                 count24 += 1
                #             elif 0.6 >= batchProbs[j][row][col] > 0.4:
                #                 range46 += bce_grad_res[j][row][col]
                #                 count46 += 1
                #             elif 0.8 >= batchProbs[j][row][col] > 0.6:
                #                 range68 += bce_grad_res[j][row][col]
                #                 count68 += 1
                #             elif 1.0 >= batchProbs[j][row][col] > 0.8:
                #                 range80 += bce_grad_res[j][row][col]
                #                 count80 += 1
                #
                #     if count02 != 0:
                #         range02 /= count02
                #     if count24 != 0:
                #         range24 /= count24
                #     if count46 != 0:
                #         range46 /= count46
                #     if count68 != 0:
                #         range68 /= count68
                #     if count80 != 0:
                #         range80 /= count80
                #
                #     embedding[idxs[j]] = np.concatenate((range02, range24, range46, range68, range80))   # [num, 16 * 5]
                #     # print('embdding:', embedding.shape) (1520, 80)

            return torch.Tensor(embedding)
