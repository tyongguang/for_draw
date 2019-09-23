#%%
import tensorflow as tf
import numpy as np
import lib.show_tb as tb

#%%
# 重点， 多参数分别训练法

tf.reset_default_graph() #否则GPU内存不释放容易OOM

#初始化的值也不能太大，太大会导致loss过大，而nan
w = tf.Variable(tf.random_uniform(shape=(1, 10), minval=-1.0, maxval=1.0, name='uniform_generator'), name='w')
b = tf.Variable(tf.random_uniform(shape=(1, 10), minval=1.0, maxval=5.0, name='uniform_generator'), name='b')
y = tf.placeholder(tf.float32, [1, 10], name='y')
batch_y = np.random.rand(1, 10) * 1000
print("src_y:", batch_y)

pred = tf.add(tf.square(w, name='sqare_predict'), b, name='add_predict')
loss = tf.square(tf.subtract(pred, y, name='sub_loss'), name='square_loss')
sum_b = tf.square( tf.reduce_sum(b) ) * 10
final_loss = tf.reduce_sum(loss, name='reduce_sum_loss') + sum_b

global_step = tf.Variable(0, trainable=False, name='global_step')
starter_learning_rate = 0.0005 #一开始的时候，目标值与期望值相差巨大，lr必须比较小，否则马上就要爆掉
learning_rate = tf.train.exponential_decay(starter_learning_rate, \
    global_step, 20, 0.7, staircase=True, name='mydecay')

opt =  tf.train.GradientDescentOptimizer(learning_rate)

#第一个是梯度，第二是变量值,表示为[(grad, var)], 所以取值是为
# grad = grads_and_vars_value[0][0]
# var =  grads_and_vars_value[0][1]
grads_and_vars = opt.compute_gradients(final_loss, var_list=[w])
train_step = opt.apply_gradients(grads_and_vars, global_step=global_step)

train_b_only = tf.train.GradientDescentOptimizer(0.0005).minimize(sum_b, var_list=[b])

init  = tf.global_variables_initializer()
all_grad_vars = []
with tf.Session() as sess:
    sess.run(init)
    print("w_init:",sess.run(w))
    print("------------------")
    for i in range(200):
        sess.run(train_b_only, feed_dict={y: batch_y})
        if i % 10 == 0:
            print("sum_b:", sess.run(sum_b))
            print("w:", sess.run(w))
    print("============================================")

    for i in range(200):
        sess.run(train_step, feed_dict={y: batch_y})
        if i % 10 == 0:
            print("step:", sess.run(global_step))
            print("lr:", sess.run(learning_rate))
            #print("first grad:", sess.run("gradients/reduce_sum_loss_grad/Tile:0", feed_dict={y: batch_y}))
            all_grad_vars.append(sess.run(grads_and_vars, feed_dict={y: batch_y}))
            print("loss:",sess.run([final_loss], feed_dict={y: batch_y}))
            print("w:", sess.run(w))
            print("b, sum_b", sess.run([b, sum_b]))
            print("------------------")
    
    #grads_w = [x[0][0] for x in all_grad_vars]
    #grads_b = [x[1][0] for x in all_grad_vars]
    
    #print(grads)
    #writer = tf.summary.FileWriter("./test", sess.graph) 
    #tb.show_default_graph()



#%%
