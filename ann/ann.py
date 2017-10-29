import numpy as np
import random
import time

def tanh(x):
    return np.tanh(x)


def tanh_derive(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derive(z):
    return sigmoid(z)*(1 - sigmoid(z))


class SpamNeuralNetwork:
    def __init__(self, layers, activation="tanh"):

        if activation == "tanh":
            self.active_function = tanh
            self.active_derive = tanh_derive
        elif activation == "sigmoid":
            self.active_function = sigmoid
            self.active_derive = sigmoid_derive
        # elif activation == "leaky_relu":
        #     self.active_function = leaky_relu
        #     self.active_derive = leaky_relu_derive

        self.layers = layers
        self.bias = [np.random.randn(x, 1) for x in layers[1:]]
        self.weight = [np.random.randn(x, y) for x, y in zip(layers[1:], layers[:-1])]

    def feed_forward(self, x):
        x.shape = [len(x), 1]
        for w, b in zip(self.weight, self.bias):
            x = self.active_function(np.dot(w, x) + b)
        return x

    def fit(self, x, y, mini_batch_size, eta, epochs, test_x, test_y):
        if test_y is not None:
            n_test = len(test_y)
        n = x.shape[0]
        start = time.time()
        for i in range(epochs):
            batch = [(x, y) for x, y in zip(x, y)]
            random.shuffle(batch)
            # mini_batches = [batch[0:mini_batch_size]]
            mini_batches = [batch[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if i % 10 == 0:
                if test_y is not None:
                    result = self.evaluate(test_x, test_y)
                    print("Epoch {0}: {1} / {2}, {3}, 用时: {4}s".format(
                        i, result, n_test, result / n_test, time.time() - start))
                    start = time.time()
                else:
                    print("Epoch {0} complete".format(i))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weight = [w - (eta / len(mini_batch)) * nw
                       for w, nw in zip(self.weight, nabla_w)]
        self.bias = [b - (eta / len(mini_batch)) * nb
                     for b, nb in zip(self.bias, nabla_b)]

    def back_prop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        zs = []
        activations = []
        x.shape = [x.shape[0], 1]
        # x = x.reshape(len(x), 1)
        activation = x
        activations.append(activation)
        for w, b in zip(self.weight, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.active_function(z)
            activations.append(activation)
        delta = self.cost_derive(activations[-1], y) * self.active_derive(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, len(self.layers)):
            delta = np.dot(self.weight[-l + 1].transpose(), delta) * self.active_derive(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def cost_derive(self, o, y):
        return o - y.reshape([len(y), 1])

    def evaluate(self, test_x, test_y):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in zip(test_x, test_y)]
        return sum(int(x == y) for (x, y) in test_results)


def main():
    data = np.loadtxt("../data/spambase.data", delimiter=',')
    np.random.shuffle(data)
    print(data.shape)

    train_data = data[:4000, :]
    valid_data = data[4000:, :]

    y = []
    for data in train_data:
        if data[-1] == 0:
            y.append(np.array([1, 0]))
        else:
            y.append(np.array([0, 1]))

    # train_data = [(x, y) for x, y in zip(train_data[:, :-1], train_data[:, -1])]
    # valid_data = [(x, y) for x, y in zip(valid_data[:, :-1], valid_data[:, -1])]

    nn = SpamNeuralNetwork([57, 48, 48, 2], activation="tanh")
    nn.fit(x=train_data[:, :-1],
           y=y,
           test_x=valid_data[:, :-1],
           test_y=valid_data[:, -1].astype(np.int),
           epochs=100000,
           mini_batch_size=4000,
           eta=0.0005)


if __name__ == '__main__':
    main()
