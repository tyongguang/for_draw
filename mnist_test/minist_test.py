#%%
# 参考：
# https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py
# https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data


#%%
mnist = input_data.read_data_sets('mnist_test/MNIST_data', one_hot=True)

#%%
def display_digit(num, images, labels):
    print(labels[num])
    label = labels[num].argmax(axis=0)
    image = images[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop, input_images):
    images = input_images[start].reshape([28,28])
    for i in range(start+1,stop):
        images = np.concatenate((images, input_images[i].reshape([28,28])), axis=1)
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()
#%%
#test display_digit
#随机选一个数字显示
display_digit(random.randint(0, len(mnist.train.images)), mnist.train.images, mnist.train.labels)
display_mult_flat(0, 5,  mnist.train.images)
#%%
#测试显示张图片，下面分别是train 和test的数据
plt.imshow(mnist.train.images[10].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
plt.show()
plt.imshow(mnist.test.images[7].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
plt.show()

#%%
# tensorboar 记录图片的变化
input = tf.placeholder(tf.float32, [None, 784], name='input')
image_shaped_input = tf.reshape(input,[-1, 28,28, 1])
img_summ = tf.summary.image('input',image_shaped_input, max_outputs = 9)
writer = tf.summary.FileWriter("./logs")
with tf.Session() as sess:
    for i in range(50):
        summ = sess.run(img_summ, feed_dict = {input: mnist.train.next_batch(20)[0]})
        writer.add_summary(summ, i)


#%%
