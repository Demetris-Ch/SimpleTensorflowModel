import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split


# Create a dataset of 1000 samples. The true coefficients used
# are a=2, b=1, c=0.
rs = np.random.RandomState(627372)
n = 1000
X = rs.randn(n, 3)
Y = ((2 * X[:, 0] + X[:, 1]) ** 3 + rs.randn(n))[:, np.newaxis]
#(a * x_1 + b * x_2 + c * x_3) ^ 3
# The task is to build a model and train it with this toy data.
# Please validate the training on a small fraction of the data.

errors = []
alpha = 0.02

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_ph = tf.placeholder(tf.float32, shape=[None, 3], name='x-input')

#Y_ph = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')
Y_ph = tf.placeholder("float")
a = tf.Variable(0.1*tf.ones([3, 1]))
#b = tf.Variable(tf.zeros([1, 1]))
#c = tf.Variable(tf.zeros([1, 1]))

model = tf.pow(tf.matmul(X_ph, a), 3)

#model = tf.pow(tf.matmul(X[:,0], a) + tf.matmul(X[:,], b) + tf.matmul(tf.pow(X, 3), c), 3)

cost = tf.reduce_sum(tf.square(Y_ph-model))/(2*n)

#optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
optimizer = tf.train.AdamOptimizer(alpha).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50000):
        sess.run(optimizer, feed_dict={X_ph: x_train, Y_ph: y_train})
        if i % 10 == 0:
            loss = sess.run(cost, feed_dict={X_ph: x_train, Y_ph: y_train})
            print("Epoch", (i + 1), ": Training Cost:", loss, " a,b,c:", sess.run(a))
            errors.append(loss)

    test_loss = sess.run(cost, feed_dict={X_ph: x_test, Y_ph: y_test})
    a = sess.run(a)

print(f"Final training error is {errors[-1]}")
print(f"Final test error is {test_loss}")
print(f"Fitted Coefficients are {a}")