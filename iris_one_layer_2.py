
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")
np.random.shuffle(data)
x_data = data[:,0:4].astype('f4')
y_data = one_hot(data[:,4].astype(int), 3)

print y_data

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])
y_ = tf.placeholder("float", [None, 3])

#Primera capa
#W (4 datos por 3 neuronas)
W = tf.Variable(np.float32(np.random.rand(4, 3))*0.1)
b = tf.Variable(np.float32(np.random.rand(3))*0.1)
#resultado primera capa
r1 = tf.nn.sigmoid(tf.matmul(x, W) + b)

#Segunda capa
#W2 (3 datos por 3 neuronas)
W2 = tf.Variable(np.float32(np.random.rand(3, 3))*0.1)
b2 = tf.Variable(np.float32(np.random.rand(3))*0.1)
#resultado segunda capa
r2 = tf.matmul(r1, W2) + b2
#normalizacion de la salida
y = tf.nn.softmax(r2)
#El error
cross_entropy = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

err = []

for step in xrange(1000):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj*batch_size : jj*batch_size+batch_size]
        batch_ys = y_data[jj*batch_size : jj*batch_size+batch_size]

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        if step % 50 == 0:
            err.append(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
            # print "Iteration #:", step, "Error: ", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            print "Iteration #:", step, "Error: ", err[-1]
            # err.append(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
            result = sess.run(y, feed_dict={x: batch_xs})
            for b, r in zip(batch_ys, result):
                print b, "-->", r
            print "----------------------------------------------------------------------------------"

print err
draw = plt.plot(err, label='Error')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()

