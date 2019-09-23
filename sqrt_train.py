#%%
import tensorflow as tf
import numpy as np
import lib.show_tb as tb

#%%
tf.reset_default_graph() #否则GPU内存不释放容易OOM

#初始化的值也不能太大，太大会导致loss过大，而nan
w = tf.Variable(tf.random_uniform(shape=(1, 10), minval=-1.0, maxval=1.0, name='uniform_generator'), name='w')
#b = tf.Variable(tf.random_uniform(shape=(1, 10), minval=0, maxval=100, name='uniform_generator'), name='b')
y = tf.placeholder(tf.float32, [1, 10], name='y')
batch_y = np.random.rand(1, 10) * 1000
print("src_y:", batch_y)
tf.summary.histogram("w", w)
#pred = tf.add(tf.square(w, name='sqare_predict'), b, name='add_predict')
pred = tf.square(w, name='sqare_predict')
loss = tf.square(tf.subtract(pred, y, name='sub_loss'), name='square_loss')
#loss = tf.abs(tf.subtract(pred, y, name='sub_loss'), name='abs_loss')
final_loss = tf.reduce_sum(loss, name='reduce_sum_loss')
tf.summary.scalar("loss", final_loss)

global_step = tf.Variable(0, trainable=False, name='global_step')
starter_learning_rate = 0.0005 #一开始的时候，目标值与期望值相差巨大，lr必须比较小，否则马上就要爆掉
learning_rate = tf.train.exponential_decay(starter_learning_rate, \
    global_step, 20, 0.7, staircase=True, name='mydecay')

opt =  tf.train.GradientDescentOptimizer(learning_rate)


#第一个是梯度，第二是变量值,表示为[(grad, var)], 所以取值是为
# grad = grads_and_vars_value[0][0]
# var =  grads_and_vars_value[0][1]
grads_and_vars = opt.compute_gradients(final_loss)
grad_w = grads_and_vars[0][0]
tf.summary.histogram("grad_w", grad_w)

train_step = opt.apply_gradients(grads_and_vars, global_step=global_step)

#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(final_loss, global_step=global_step) 
init  = tf.global_variables_initializer()
all_grad = []

summary_merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", tf.get_default_graph()) 


with tf.Session() as sess:
    sess.run(init)
    print("w_init:",sess.run(w))
    print("------------------")
    for i in range(100):
        _, np_summary_merged, np_global_step = sess.run([train_step, summary_merged, global_step], feed_dict={y: batch_y})
        writer.add_summary(np_summary_merged, np_global_step)
        if i % 10 == 0:
            np_final_loss , np_grad_w, np_learning_rate, np_w = \
                sess.run([final_loss, grad_w, learning_rate, w], 
                feed_dict={y: batch_y})

            print("step:%d, lr:%f, loss:%f" % (np_global_step, np_learning_rate, np_final_loss) )
            print("w:", np_w)
            
            all_grad.append(np_grad_w)
            

            #print("b:", sess.run(b))
            print("------------------")
    
        
  

    #tb.show_default_graph()



#%%
