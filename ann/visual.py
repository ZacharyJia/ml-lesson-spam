import numpy as np
from matplotlib import pyplot
import re


# def main():
#     with open("../data/train_process.data", encoding="utf-8") as f:
#         l = []
#         line = f.readline()
#         while line:
#             data = re.findall(r"Epoch (\d+): \d+ / \d+, (0.\d+)", line)[0]
#             data = np.array([int(data[0]), float(data[1])])
#             l.append(data)
#             line = f.readline()
#         l = np.array(l)
#         print(l)
#         np.save("../data/train_process", l)
def main():
    l = []
    with open("../data/train_process2.data", encoding="utf-8") as f:
        line = f.readline()
        while line:
            data = re.findall(r"Epoch (\d+): \d+ / \d+, (0.\d+)", line)[0]
            data = np.array([int(data[0]), float(data[1])])
            l.append(data)
            line = f.readline()
    data = np.array(l)

    # data = np.load("../data/train_process.npy")
    z = np.polyfit(data[:, 0], data[:, 1], 10)
    p1 = np.poly1d(z)
    yvals = p1(data[:, 0])
    print(p1)
    pyplot.plot(data[:, 0], data[:, 1], linewidth=1)
    pyplot.plot(data[:, 0], yvals, 'r')
    pyplot.show()


if __name__ == '__main__':
    main()
