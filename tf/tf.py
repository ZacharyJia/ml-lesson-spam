import tensorflow as tf
import numpy as np
import random

data = np.loadtxt("../data/spambase.data", delimiter=',', dtype=np.float32)
np.random.shuffle(data)

train_data = data[:4000, :]
valid_data = data[4000:, :]

# y = train_data[:, -1]
y = []
x = train_data[:, :-1]
for data in train_data:
    if data[-1] == 0:
        y.append(np.array([1, 0]))
    else:
        y.append(np.array([0, 1]))

valid_y = []
for data in valid_data:
    if data[-1] == 0:
        valid_y.append(np.array([1, 0]))
    else:
        valid_y.append(np.array([0, 1]))
valid_y = np.array(valid_y)

y = np.array(y)

graph = tf.get_default_graph()
writer = tf.summary.FileWriter("log", graph)


def next_batch(i, size):
    np.random.shuffle([x, y])

    return x[0:size].reshape([-1, 57]), y[0:size].reshape([-1, 2])


x1 = tf.placeholder(tf.float32, [None, 57])
W1 = tf.Variable(tf.zeros([57, 48]))
b1 = tf.Variable(tf.zeros([48]))

y1 = tf.sigmoid(tf.matmul(x1, W1) + b1)

W2 = tf.Variable(tf.zeros([48, 48]))
b2 = tf.Variable(tf.zeros([48]))

y2 = tf.sigmoid(tf.matmul(y1, W2) + b2)

W3 = tf.Variable(tf.zeros([48, 2]))
b3 = tf.Variable(tf.zeros([2]))


# y = tf.sigmoid(tf.matmul(y1, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 2])

output = tf.sigmoid(tf.matmul(y2, W3) + b3)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)), reduction_indices=[1])

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# for value in [cross_entropy]:
#     tf.summary.scalar(value.op.name, value)
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summaries = tf.summary.merge_all()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(1000000):
    batch_xs, batch_ys = next_batch(i, 4000)
    sess.run(train_step, feed_dict={x1: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        writer.add_summary(sess.run(summaries, feed_dict={x1: valid_data[:, :-1], y_: valid_y}), i)
        writer.flush()
        print("{0}: accuracy: {1}".format(i, sess.run(accuracy, feed_dict={x1: valid_data[:, :-1], y_: valid_y})))
        # print(sess.run(cross_entropy, feed_dict={x1: batch_xs, y_: batch_ys}))
        # print(sess.run(output, feed_dict={x1: valid_data[:, :-1], y_: valid_y}))
