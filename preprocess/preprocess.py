import numpy as np


def main():
    data = np.loadtxt("../data/spambase.data", delimiter=',')
    data = np.where(data != 0, 1, 0).astype(np.int)
    np.save("../data/spambase", data)


if __name__ == "__main__":
    main()