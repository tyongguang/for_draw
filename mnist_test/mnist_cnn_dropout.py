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

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
labels = tf.placeholder(tf.float32, [None, 10], name='y-input')
drop_rate = tf.placeholder_with_default(0.0, shape=())


with tf.name_scope("main"):
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("layer1"):
        filter1 = tf.Variable(tf.truncated_normal((5, 5, 1, 16)), name="filter1")
        bias1 = tf.Variable(np.ones((1, 16))* 0.1, dtype=tf.float32, name="cnn_bias1")
        conv2d_output1 = tf.nn.conv2d(inputs, filter1, [1, 1, 1, 1], "VALID") + bias1
        pooling_output1 = tf.nn.max_pool(conv2d_output1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID" )  
        relu_output1 = tf.nn.relu(pooling_output1)
        relu_output1 = tf.nn.dropout(relu_output1, rate = drop_rate)

    with tf.name_scope("layer2"):
        filter2 = tf.Variable(tf.truncated_normal((3, 3, 16, 32)), name="filter2")
        bias2 = tf.Variable(np.ones((1, 32))* 0.1 , dtype=tf.float32, name="cnn_bias2")
        conv2d_output2 = tf.nn.conv2d(relu_output1, filter2, [1, 1, 1, 1], "VALID") + bias2
        pooling_output2 = tf.nn.max_pool(conv2d_output2, [1, 2, 2, 1], [1, 2, 2, 1] , padding="VALID")  
        relu_output2 = tf.nn.relu(pooling_output2)
        relu_output2 = tf.nn.dropout(relu_output2, rate = drop_rate)

    with tf.name_scope("flatten"):
        flatten_output = tf.layers.flatten(relu_output2)

    with tf.name_scope("fc1"):
        fc_w1 = tf.Variable(tf.truncated_normal((800, 512)), name="w1")
        fc_b1 = tf.Variable(np.ones((1, 512))* 0.1 , dtype=tf.float32, name="fc_b1")
        fc_ouptut1 = tf.nn.relu(tf.matmul(flatten_output, fc_w1) + fc_b1)

    with tf.name_scope("fc2"):
        fc_w2 = tf.Variable(tf.truncated_normal((512, 10)), name="w2")
        fc_b2 = tf.Variable(np.ones((1, 10))* 0.1 , dtype=tf.float32, name="fc_b2")
        fc_ouptut2 = tf.matmul(fc_ouptut1, fc_w2) + fc_b2

    with tf.name_scope("softmax"):
        predict = tf.nn.softmax(fc_ouptut2)

    with tf.name_scope("metrics"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_ouptut2, labels=labels), name="loss")
        acc = util.accuracy(predict, labels)

    with tf.name_scope("main_train"):
        opt = tf.train.AdamOptimizer(0.0001)
        grad_and_var = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grad_and_var, global_step = global_step)

gen_ops = []
gen_loss_op = []
with tf.name_scope("inspect"):
    gen_image = tf.Variable(tf.truncated_normal((1, 28, 28, 1)), name="gen_image")
    conv2d_output_x = tf.nn.conv2d(gen_image, filter1, [1, 1, 1, 1], "VALID") + bias1
    pooling_output_x = tf.nn.max_pool(conv2d_output_x, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID" )  
    relu_output_x = tf.nn.relu(pooling_output_x)
    
    filter_result1, _ = tf.split(relu_output_x, [1, 15], -1)
    filter_sum = -1 * tf.reduce_sum(filter_result1) + tf.reduce_sum(tf.square(gen_image)) * 2
    gen_image_step = tf.train.AdamOptimizer(0.0001).minimize(filter_sum, var_list=gen_image)
    gen_ops.append(gen_image_step)
    gen_loss_op.append(filter_sum)
    
    for d in range(9):
        _, filter_result_d, _ = tf.split(relu_output_x, [d + 1, 1, 16 - d -2], -1)
        filter_sum_d = -1 * tf.reduce_sum(filter_result_d) + tf.reduce_sum(tf.square(gen_image)) * 2
        gen_ops.append(tf.train.AdamOptimizer(0.0001).minimize(filter_sum_d, var_list=gen_image))
        gen_loss_op.append(filter_sum_d)

# create saver
saver = util.TrainSaver("ckpt_zoo/mnist_cnn_dropout/cnn_dropout.ckpt", "main")

#%%
# train !
training_grads = util.get_grads(grad_and_var)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.try_load(sess)
    saver.remove_old_ckpt(sess, global_step=global_step)
        
    for i in range(TRAIN_STEPS):
        train_images, train_labels = mnist.train.next_batch(BATCH_SIZE)
        train_images =  train_images.reshape(-1, 28, 28, 1)
        _, np_loss = sess.run([train_step, loss], feed_dict={inputs: train_images, labels: train_labels, drop_rate:0.5})

        if i % (TRAIN_STEPS/10) == 0:
            loss_value, step, acc_value, grad_list = sess.run([loss, global_step, acc, training_grads], feed_dict={inputs: train_images, labels: train_labels, drop_rate:0.5})
            print("After %d training step(s), loss: %g, acc:%f." % (step, loss_value, acc_value))
            saver.save(sess, global_step=global_step)

            test_accuracy = sess.run(acc, feed_dict={inputs: mnist.test.images.reshape(-1, 28, 28, 1),
             labels: mnist.test.labels})
            print("test accuracy:", test_accuracy)
            all_grad = np.empty((0)).flatten()
            for g in grad_list:
                all_grad = np.append(all_grad, g.flatten())

            plt.title("test accuracy:%s" % test_accuracy)
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

    x, y = mnist.test.next_batch(1)
    x = x.reshape(-1, 28, 28 , 1)
    value, np_relu_output1 = sess.run([predict, relu_output1], 
        feed_dict = {inputs:x })
    print("predict: %d, raw_vector:%s" % (np.argmax(value), value))
    print("label:", np.argmax(y))
    plt.imshow(x.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.show()

#%%
# 查看错误的CASE
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if False == saver.try_load(sess):
        raise Exception("not train?")

    x = mnist.test.images
    y = mnist.test.labels
    x = x.reshape(-1, 28, 28 , 1)
    predict_value, fc_output = sess.run([predict, fc_ouptut2], feed_dict = {inputs:x })
    index_predictions = np.argmax(predict_value, -1)
    index_labels = np.argmax(y, -1)
    diff  = index_labels - index_predictions
    err_index = np.where(diff  != 0)

    # diff.where(())
    for i in err_index[0]:
        plt.title('%d  Label: %d, predict:%d, raw:%s ' % (i,  np.argmax(y[i]), 
            np.argmax(predict_value[i]), np.around(predict_value[i], decimals=2) ))
        plt.imshow(x[i].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
        plt.show()
        if i >1500: #查看前1500号有一些
            break


#%%
#查看参数和grads
with tf.Session() as sess:
    if False == saver.try_load(sess):
        raise Exception("not train?")

    vars = sess.run(util.get_vars(grad_and_var))

    train_images, train_labels = mnist.train.next_batch(BATCH_SIZE)
    train_images =  train_images.reshape(-1, 28, 28, 1)
    grads = sess.run(util.get_grads(grad_and_var), feed_dict={inputs: train_images, labels: train_labels})

    names = util.get_names(grad_and_var)

    for v, name in zip(vars, names):
        plt.title("var:[%s]" %  name)
        plt.hist(v.flatten(), bins=50)
        plt.show()
    print("----------------------")
    for g, name in zip(grads, names):
        plt.title("grad:[%s]" % name)
        plt.hist(g.flatten(), bins=50)
        plt.show()
    
    

#%%
# 生成一张图片，使卷积结果最大化，也就是跟filter匹配得最好的图片
# index 是 第一层filter 第2个filter(从0开始)
gen_steps = 200000
index = 2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if False == saver.try_load(sess):
        raise Exception("not train?")


    pre_loss = 1e9
    for i in range(gen_steps):
        sess.run(gen_ops[0])
        if i % (gen_steps / 20) == 0:
            np_relu_output_x, gen_image_result, gen_loss = sess.run([relu_output_x, gen_image, gen_loss_op[index]])
            gen_image_loss = np_relu_output_x[:,:,:, index]
            print("lost:", gen_loss)
            if math.fabs(gen_loss - pre_loss) < 0.01:
                break            
            if gen_loss < pre_loss:
                pre_loss = gen_loss

            plt.imshow(gen_image_result.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
            plt.show()

#%%
# 上面的测试用例正式化，下面生成0 ~ 8 号filter 匹配最好的图片
def gen_filter_image(index):
    gen_steps = 200000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if False == saver.try_load(sess):
            raise Exception("not train?")


        pre_loss = 1e9
        for i in range(gen_steps):
            sess.run(gen_ops[0])
            if i % (gen_steps / 20) == 0:
                np_relu_output_x, gen_image_result, gen_loss = sess.run([relu_output_x, gen_image, gen_loss_op[index]])
                gen_image_loss = np_relu_output_x[:,:,:, index]
                print("lost:", gen_loss)
                if math.fabs(gen_loss - pre_loss) < 0.01:
                    break            
                
                pre_loss = gen_loss

        plt.imshow(gen_image_result.reshape(28, 28), cmap=plt.get_cmap('gray_r'))
        plt.show()
for i in range(9):
    gen_filter_image(i)

#%%
#save dropout model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if False == saver.try_load(sess):
        raise Exception("not train?")

    
    target_def = util.save_as_pb("cnn_drop_out.pb", sess, ["x-input"], ["main/softmax/Softmax"])


#%%
