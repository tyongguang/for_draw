#%%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import lib.show_tb as tb

    
#%%
mnist = input_data.read_data_sets("mnist_test/MNIST_data", one_hot=True)

#%%
#train
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="mnist_test/MNIST_model/"
MODEL_NAME="mnist_model"

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

tf.reset_default_graph()

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
y = inference(x, regularizer)
global_step = tf.Variable(0, trainable=False)


# variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# variables_averages_op = variable_averages.apply(tf.trainable_variables())
sparse_labels = tf.argmax(y_, axis=-1)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=sparse_labels)
prediction =  tf.nn.softmax(y)
prediction_index = tf.argmax(prediction, axis=-1)
equality = tf.math.equal(sparse_labels, prediction_index)
acc =  tf.reduce_mean(tf.cast(equality, tf.float32)) 

cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
    staircase=True)
opt = tf.train.GradientDescentOptimizer(learning_rate)
grad_and_var = opt.compute_gradients(loss)
for grad, var in grad_and_var: 
    tf.summary.histogram(var.name, grad)

merged_summ = tf.summary.merge_all()

writer = tf.summary.FileWriter("./logs")
train_op = opt.apply_gradients(grad_and_var, global_step=global_step)

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(TRAINING_STEPS):
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        _, byte_summ = sess.run([train_op, merged_summ], feed_dict={x: xs, y_: ys})
        writer.add_summary(byte_summ, global_step=i) #记录梯度的变化
        if i % 100 == 0:
            loss_value, step, accuracy = sess.run([loss, global_step, acc], feed_dict={x: xs, y_: ys})
            print("After %d training step(s), loss: %g, acc:%f." % (step, loss_value, accuracy))
            #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            test_accuracy = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("test accuracy:", test_accuracy)



#%%
