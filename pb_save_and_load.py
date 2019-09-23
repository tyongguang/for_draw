#%%
import numpy as np
import tensorflow as tf
import lib.show_tb as tb
import lib.util as util

data = np.random.rand(1, 7, 7, 1024)

#%%
#保存
tf.reset_default_graph()
x = tf.placeholder(shape=(None, 7, 7, 1024), dtype=tf.float32, name="input")
mean_op = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
pool_op = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID', name="mypool")

with tf.Session() as sess:
    mean_result, pool_result = sess.run([mean_op, pool_op], feed_dict={x: data})
    xx_def = util.save_as_pb("xx.pb", sess, ["input"], ["mypool"])


np_result = np.mean(np.mean(data, axis=1), axis=1).reshape(1, 1, 1, 1024)
#%%
#恢复。在运行第一个block后，可以不运行上面的block，而直接运行此block
tf.reset_default_graph()
util.import_pb("xx.pb")
with tf.Session() as sess:
    result = sess.run("mypool:0", feed_dict = {"input:0": data })
    print(result)

#%%
