
#%%
#!/usr/bin/python
# coding:utf-8

# TensorBoard直方图仪表板
import tensorflow as tf

#%%
tf.reset_default_graph()
k = tf.placeholder(tf.float32)
# 创建一个均值变化的正态分布（由0到5左右）
mean_moving_normal = tf.random_normal(shape=[100, 100, 2], mean=(9*k), stddev=1)
# 将该分布记录到直方图汇总中
histogram_summ = tf.summary.histogram("normal/moving_mean", mean_moving_normal)
sess = tf.Session()
writer = tf.summary.FileWriter("./logs/")
summaries = tf.summary.merge_all()
# 设置一个循环并将摘要写入磁盘
N = 400
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    h_summ = sess.run(histogram_summ, feed_dict={k: k_val})
    #writer.add_summary(summ, global_step=step)
    writer.add_summary(h_summ, global_step=step)

#%%
