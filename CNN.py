import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# downloadimport tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#(55000 photo 28*28 pixel)
# training data
mnist= input_data.read_data_sets('minist_data', one_hot=True)
# one_hot is a kind of encoding pattern
# 0, 1,2,3,4,5,6,7,8,9
# 0:1000000000
# 1:0100000000
# 2:0010000000
# Test data set
text_x=mnist.test.images[:3000]
text_y=mnist.test.labels[:3000]

# None the first dimension of the tensor
input_x =tf.placeholder(tf.float32,[None,28*28])/255.
output_y =tf.placeholder(tf.int32,[None,10])
input_x_images=tf.reshape(input_x,[-1,28,28,1])

# the first conv layer
conv1= tf.layers.conv2d(
        inputs=input_x_images,
        filters=32, # 32个过滤器
        kernel_size=[5,5], #过滤器大小
        strides=1, # step size
        padding='same', # 在外围补0，输出大小不变)
        activation=tf.nn.relu)

# output 28*28*3
# the first pooling layer (pooling just like line pooling)
pool1=tf.layers.max_pooling2d(
        inputs=conv1, # 28*28*32
        pool_size=[2,2],
        strides=2)
# 14*14*32

# the second conv layer
conv2= tf.layers.conv2d(
        inputs=pool1, #14*14*32
        filters=64, # 64个过滤器
        kernel_size=[5,5], #过滤器大小
        strides=1, # step size
        padding='same', # 在外围补0，输出大小不变)
        activation=tf.nn.relu)
# 14*14*64

# the second pooling layer (pooling just like line pooling)
pool2=tf.layers.max_pooling2d(
        inputs=conv2, # 14*14*64
        pool_size=[2,2],
        strides=2)
# 7*7*64

# flat
flat=tf.reshape(pool2,[-1,7*7*64])

dense=tf.layers.dense(inputs=flat,units=1024, activation=tf.nn.relu)
# Dropout
dropout=tf.layers.dropout(inputs=dense,rate=0.5,training=True)

# 10 neurons dense layer without activation(dense全连接)
logits =tf.layers.dense(inputs=dropout,units=10)

loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)

# Adam
train_op=tf.train.AdamOptimizer(0.001).minimize(loss)

accuracy =tf.metrics.accuracy(
        labels=tf.argmax(output_y,axis=1),
        predictions=tf.argmax(logits,axis=1),)[1]


with tf.Session() as sess:
        init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init)
        for  i in range(300):
                batch=mnist.train.next_batch(50)
                train_loss,train_op_1=sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
                if i%1==0:
                        test_accuracy=sess.run(accuracy,{input_x:text_x,output_y:text_y})
                        print("Step=%d, Train loss=%4f, [Test accur=%.2f" %(i, train_loss,test_accuracy))

        test_output=sess.run(logits,{input_x:text_x[:20]})
        inferenced_y =np.argmax(test_output,1)
        print(inferenced_y,'inferenced numbers')
        print(np.argmax(text_y[:20],1),'Real numbers')




