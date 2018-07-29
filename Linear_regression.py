import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''
Linear regression using gradient descent
'''

# [1] input data
points_num=100
vector=[]

for i in range(points_num):
    x1=np.random.normal(0.0,0.66)
    y1=0.1*x1+0.2+np.random.normal(0.0,0.04)
    vector.append([x1,y1])

x_data=[v[0] for v in vector]
y_data = [v[1] for v in vector]


# [2] Fig 1
plt.plot(x_data,y_data,'r*',label="Original data")
plt.title("Linear regression")
plt.legend()
plt.show()

# Graph
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))
y=W*x_data+b

# loss function
loss=tf.reduce_mean(tf.square(y-y_data))

# Gradient
optimizer =tf.train.GradientDescentOptimizer(0.5)# Learning rate
train =optimizer.minimize(loss)
# session
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for step in range(20):
        sess.run(train)
        print("step=%d,loss=%f,[Weight=%f Bias=%f]"  %(step, sess.run(loss), sess.run(W),sess.run(b)))

    plt.plot(x_data, y_data, 'r*', label="Original data")
    plt.title("Linear regression")
    plt.plot(x_data,sess.run(W)*x_data+sess.run(b),label="Fitted line")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



