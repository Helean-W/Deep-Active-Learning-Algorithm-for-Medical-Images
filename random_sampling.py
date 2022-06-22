import numpy as np

class Strategy_Random:
    def __init__(self, X, Y, idxs_lb):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.n_pool = len(Y)

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb


class RandomSamplingUnext(Strategy_Random):
    def __init__(self, X, Y, idxs_lb):
        super(RandomSamplingUnext, self).__init__(X, Y, idxs_lb)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        np.random.shuffle(idxs_unlabeled)
        return idxs_unlabeled[:n]