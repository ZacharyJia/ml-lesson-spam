import numpy as np
import math
from sklearn import naive_bayes


class SpamGaussianNB:
    p_h = None
    p_s = None
    mean_s = None
    mean_h = None
    var_s = None
    var_h = None

    @staticmethod
    def _gaussian_distribute(mean, var, x):
        if var == 0:
            if x == mean:
                return 1
            else:
                return 0
        else:
            return 1.0 / math.sqrt(2 * math.pi * var) * math.exp(-math.pow(x - mean, 2) / (2 * var))

    def fit(self, x, y):
        n = x.shape[0]
        n_s = np.count_nonzero(y)
        n_h = n - n_s

        self.p_h = n_h / n
        self.p_s = 1 - self.p_h

        self.mean_s = np.zeros(x[0].size)
        self.mean_h = np.zeros(x[0].size)

        self.var_s = np.zeros(x[0].size)
        self.var_h = np.zeros(x[0].size)

        for i in range(len(y)):
            if y[i] == 1:
                self.mean_s += x[i]
            if y[i] == 0:
                self.mean_h += x[i]
        self.mean_s = self.mean_s / n_s
        self.mean_h = self.mean_h / n_h

        for i in range(len(y)):
            if y[i] == 1:
                self.var_s += (x[i] - self.mean_s) * (x[i] - self.mean_s).transpose()
            if y[i] == 0:
                self.var_h += (x[i] - self.mean_h) * (x[i] - self.mean_h).transpose()
        self.var_s = self.var_s / n_s
        self.var_h = self.var_h / n_h

    def predict(self, x):
        if self.p_s is None:
            raise Exception("Must fit first!!")
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            h_s = 1
            h_h = 1
            for j in range(x[i].size):
                h_s *= self._gaussian_distribute(self.mean_s[j], self.var_s[j], x[i][j])
                h_h *= self._gaussian_distribute(self.mean_h[j], self.var_h[j], x[i][j])
            result[i] = np.argmax(np.array([h_h * self.p_h, h_s * self.p_s]))
        return result


def main():
    data = np.loadtxt("../data/spambase.data", delimiter=',')
    np.random.shuffle(data)
    print(data.shape)

    train_data = data[:3000, :]
    valid_data = data[3000:, :]
    print(train_data.shape)
    print(valid_data.shape)

    sgbn = SpamGaussianNB()
    sgbn.fit(train_data[:, :-1], train_data[:, -1])
    res = sgbn.predict(valid_data[:, :-1])
    print((valid_data.shape[0] - np.count_nonzero(res - valid_data[:, -1])) / valid_data.shape[0])

    clf = naive_bayes.GaussianNB()
    clf.fit(train_data[:, :-1], train_data[:, -1])
    res = clf.predict(valid_data[:, :-1])
    print((valid_data.shape[0] - np.count_nonzero(res - valid_data[:, -1])) / valid_data.shape[0])


if __name__ == '__main__':
    main()
