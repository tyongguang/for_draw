#%%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import lib.show_tb as tb
import lib.util as util
import numpy as np
import matplotlib.pyplot as plt
import math

#%%
mnist = input_data.read_data_sets("mnist_test/MNIST_data", one_hot=True)
BATCH_SIZE = 1000
TRAIN_STEPS = 10000

#%%
#define model

tf.reset_default_graph()
in_size = 784
h1_size = 500
h2_size = 256
h3_size = 128
h4_size = 2

input = tf.placeholder(tf.float32, [None, in_size], name='input')
encoder_input = tf.placeholder(tf.float32, [None, h4_size], name='encoder_input')

with tf.name_scope("main") as main_scope:
    global_step = tf.Variable(0, trainable=False)
    eb1 = tf.Variable(tf.truncated_normal((1, h1_size)), name="eb1")
    eb2 = tf.Variable(tf.truncated_normal((1, h2_size)), name="eb2")
    eb3 = tf.Variable(tf.truncated_normal((1, h3_size)), name="eb3")
    eb4 = tf.Variable(tf.truncated_normal((1, h4_size)), name="eb4")

    e1 = tf.Variable(tf.truncated_normal((in_size, h1_size)), name="e1")
    e2 = tf.Variable(tf.truncated_normal((h1_size, h2_size)), name="e2")
    e3 = tf.Variable(tf.truncated_normal((h2_size, h3_size)), name="e3")
    e4 = tf.Variable(tf.truncated_normal((h3_size, h4_size)), name="e4")

    d4 = tf.Variable(tf.truncated_normal((h4_size, h3_size)), name="d4")
    d3 = tf.Variable(tf.truncated_normal((h3_size, h2_size)), name="d3")
    d2 = tf.Variable(tf.truncated_normal((h2_size, h1_size)), name="d2")
    d1 = tf.Variable(tf.truncated_normal((h1_size, in_size)), name="d1")

    db1 = tf.Variable(tf.truncated_normal((1, in_size)), name="db1")
    db2 = tf.Variable(tf.truncated_normal((1, h1_size)), name="db2")
    db3 = tf.Variable(tf.truncated_normal((1, h2_size)), name="db3")
    db4 = tf.Variable(tf.truncated_normal((1, h3_size)), name="db4")

    act = tf.nn.sigmoid
    e1_output = act(tf.matmul(input ,    e1) + eb1 )
    e2_output = act(tf.matmul(e1_output, e2) + eb2 )
    e3_output = act(tf.matmul(e2_output, e3) + eb3 )
    e4_output = act(tf.matmul(e3_output, e4) + eb4 )

    d4_output = act(tf.matmul(e4_output, d4) + db4 )
    d3_output = act(tf.matmul(d4_output, d3) + db3 )
    d2_output = act(tf.matmul(d3_output, d2) + db2 )
    loss_output = tf.matmul(d2_output, d1) + db1
    d1_output = tf.nn.sigmoid(loss_output)

    
    l1_reg = tf.reduce_mean(tf.abs(d1_output)) * 0.01


    loss = tf.losses.mean_squared_error(input, d1_output, scope=main_scope) + l1_reg
    
    inf_d4_output = act(tf.matmul(encoder_input, d4) + db4 )
    inf_d3_output = act(tf.matmul(inf_d4_output, d3) + db3 )
    inf_d2_output = act(tf.matmul(inf_d3_output, d2) + db2 )
    inf_d1_output = tf.nn.sigmoid(tf.matmul(inf_d2_output, d1) + db1 )
    output = tf.identity(inf_d1_output, 'output')

    with tf.name_scope("main_train"):
        opt = tf.train.AdamOptimizer(0.01)
        grad_and_var = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grad_and_var, global_step = global_step)


# create saver
saver = util.TrainSaver("ckpt_zoo/mnist_auto_encoder/mnist_auto_encoder.ckpt", "main")

#%%
# train !
training_grads = util.get_grads(grad_and_var)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.try_load(sess)
    saver.remove_old_ckpt(sess, global_step=global_step)
        
    for i in range(TRAIN_STEPS):
        train_images, _ = mnist.train.next_batch(BATCH_SIZE)
        _, np_loss = sess.run([train_step, loss], feed_dict={input: train_images})

        if i % (TRAIN_STEPS/10) == 0:
            loss_value, step, grad_list = sess.run([loss, global_step, training_grads], 
                feed_dict={input: train_images})
            print("After %d training step(s), loss: %g" % (step, loss_value))
            saver.save(sess, global_step=global_step)

            all_grad = np.empty((0)).flatten()
            for g in grad_list:
                all_grad = np.append(all_grad, g.flatten())

            plt.title("loss:%s" % str(loss_value))
            plt.hist(all_grad, bins=200)
            plt.show()
            if np_loss  < 0.001  and acc_value > 0.99999:
                print("train completed!")
                break


#%%
# test !
#random test some case and visualize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if False == saver.try_load(sess):
        raise Exception("not train?")

    x, _ = mnist.test.next_batch(1)
    code, np_relu_output1, renew_image = sess.run([e4_output, loss, d1_output], 
        feed_dict = {input: x })

    new_image = sess.run(output, 
        feed_dict = {encoder_input: code })     

    plt.imshow(x.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.show()
    plt.imshow(new_image.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.show()


#%%
#查看参数和grads
train_images, _ = mnist.train.next_batch(BATCH_SIZE)
util.hist_grad_vars(saver, grad_and_var, feed_dict={input: train_images})





#%%
