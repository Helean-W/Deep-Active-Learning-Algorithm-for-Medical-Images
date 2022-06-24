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


                # maxInds = topk(sigmoid(cout)) # H, W
                #
                # for idx, (x,y) in enumerate(maxInds):
                #     if batchProbs[x, y] > 0.5:
                #         embedding[][] = deepcopy(out[]) * (1 - batchProbs[])
                #     else:
                #         embedding[][] = deepcopy(out[]) * (0 - batchProbs[])

            return torch.Tensor(embedding)
