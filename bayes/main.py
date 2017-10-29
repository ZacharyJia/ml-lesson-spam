import numpy as np
from sklearn import naive_bayes


class SpamBernoulliNB:
    fai_s = None
    fai_h = None
    fai_s_minus = None
    fai_h_minus = None
    p_s = None
    p_h = None

    def fit(self, x, y):
        self.fai_s = np.zeros(x[0].size)
        self.fai_h = np.zeros(x[0].size)

        n = y.size
        n_s = np.count_nonzero(y)
        n_h = n - n_s

        self.p_s = n_s / n
        self.p_h = 1 - self.p_s

        for i in range(len(y)):
            if y[i] == 1:
                self.fai_s += x[i]
            if y[i] == 0:
                self.fai_h += x[i]

        self.fai_s = self.fai_s / n_s
        self.fai_h = self.fai_h / n_h

        ones = np.ones_like(self.fai_h)
        self.fai_s_minus = ones - self.fai_s
        self.fai_h_minus = ones - self.fai_h

    def predict(self, x):
        if self.p_s is None:
            raise Exception("Must fit first!!")
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            h_s = 1
            h_h = 1
            for j in range(x[i].size):
                if x[i][j] == 1:
                    h_s *= self.fai_s[j]
                    h_h *= self.fai_h[j]
                else:
                    h_h *= self.fai_h_minus[j]
                    h_s *= self.fai_s_minus[j]
            result[i] = np.argmax(np.array([h_h * self.p_h, h_s * self.p_s]))
        return result


def main():
    data = np.load("../data/spambase.npy")
    np.random.shuffle(data)
    print(data.shape)

    train_data = data[:3000, :]
    valid_data = data[3000:, :]

    sbnb = SpamBernoulliNB()
    sbnb.fit(train_data[:, :-1], train_data[:, -1])
    res = sbnb.predict(valid_data[:, :-1])
    print((valid_data.shape[0] - np.count_nonzero(res - valid_data[:, -1])) / valid_data.shape[0])

    clf = naive_bayes.BernoulliNB()
    clf.fit(train_data[:, :-1], train_data[:, -1])
    res = clf.predict(valid_data[:, :-1])
    print((valid_data.shape[0] - np.count_nonzero(res - valid_data[:, -1])) / valid_data.shape[0])


if __name__ == "__main__":
    main()
