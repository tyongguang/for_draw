#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.show_tb as tb
import lib.util as util


HIDDEN_SIZE = 10                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # LSTM的层数。
TIMESTEPS = 10                              # 循环神经网络的训练序列长度。这里比较坑爹，就算把它改为1，效果还是不错。
TRAINING_STEPS = 10000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
TRAINING_EXAMPLES = 10000                   # 训练数据个数。
TESTING_EXAMPLES = 1000                     # 测试数据个数。
SAMPLE_GAP = 0.01                           # 采样间隔。


#%%
def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  

# 用正弦函数生成训练和测试数据集合。
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))


#%%
tf.reset_default_graph()
input = tf.placeholder(shape=(None, 1, TIMESTEPS), dtype=tf.float32, name="input")
lable = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="label")



with tf.name_scope("LSTM_Main"):
    global_step = tf.Variable(0, trainable=False)
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) 
#        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) 
        for _ in range(NUM_LAYERS)])    

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    outputs, _ = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32, scope = "Dynamic_rnn")
    output = outputs[:, -1, :]

    # FC
    _predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None, scope="FC")
    predictions = tf.identity(_predictions, name="output")

    loss = tf.losses.mean_squared_error(labels=lable, predictions=predictions)

    opt = tf.train.AdagradOptimizer(0.01)
    grad_and_var = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grad_and_var, global_step = global_step)


saver = util.TrainSaver("ckpt_zoo/lstm_sin/lstm_sin.cpkt")
ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
train_input, train_label = ds.make_one_shot_iterator().get_next()

#%%
# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.try_load(sess)
    saver.remove_old_ckpt(sess, global_step )
    for  i in range(TRAINING_STEPS):
        train_input_data, train_label_data = sess.run([train_input, train_label])
        #print(train_input_data)
        _, loss_value, step = sess.run([train_op, loss, global_step], 
            feed_dict = {input: train_input_data, lable: train_label_data})
        if i % (TRAINING_STEPS /20) == 0:
            loss_value = sess.run([loss], 
                feed_dict = {input: train_input_data, lable: train_label_data})
            print("step:%d, loss:%s" % (step, str(loss_value)))
            saver.save(sess, global_step )

#%%
#评估模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if False == saver.try_load(sess):
        raise Exception("not train")

    list_predict_values = sess.run([predictions], 
        feed_dict = {input: test_X}) 
    predict_values  = np.asarray(list_predict_values, dtype=np.float32).reshape(*test_y.shape)

    #plt predict and test
    plt.figure()
    plt.plot(predict_values, label='predictions', linewidth=4)
    plt.plot(test_y, label='real_sin')
    plt.legend()
    plt.show()
tb.save_default_graph()

#%%
#export and test onnx file
import tf2onnx.convert as onnx_conv
import os
import onnxruntime as rt
from subprocess import call
import time

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if False == saver.try_load(sess):
        raise Exception("not train")
    
    #get tensorflow value
    start = time.time()
    for _ in range(100):
        tf_list_predict_values = sess.run([predictions], feed_dict = {input: train_X}) 
        #tf_list_predict_values = sess.run([predictions], feed_dict = {input: test_X}) 
    end = time.time()
    tf_predict_values  = np.asarray(tf_list_predict_values, dtype=np.float32).flatten()
    

    print("tf:%f" % (end -start))
    #export pb file
    lstm_sin_def = util.save_as_pb("lstm_sin.pb", sess, ["input"], ["LSTM_Main/output"])

#pb --> onnx
cmd_line = ["python3", "-m", "tf2onnx.convert", "--opset", "8", "--input", "lstm_sin.pb", "--inputs", "input:0", "--outputs", "LSTM_Main/output:0", "--output", "lstm_sin2.onnx"]
os.system(" ".join(cmd_line))

#execute onnx file
rt_sess = rt.InferenceSession("lstm_sin2.onnx")
input_name = rt_sess.get_inputs()[0].name
start = time.time()
for _ in range(100):
    onnx_list_predict_values = rt_sess.run(None, {input_name: train_X})
    #onnx_list_predict_values = rt_sess.run(None, {input_name: test_X})
end = time.time()
print("onnx:%f" % (end -start))
onnx_predict_values  = np.asarray(onnx_list_predict_values, dtype=np.float32).flatten()

#diff
plt.figure()
plt.plot(onnx_predict_values, label='tf', linewidth=4)
plt.plot(tf_predict_values, label='onnx')
plt.legend()
plt.show()


#%%
#获得参数
with tf.Session() as sess:
    if False == saver.try_load(sess):
        raise Exception("not train?")

    train_input_data, train_label_data = sess.run([train_input, train_label])

    vars, grads = sess.run([util.get_vars(grad_and_var), util.get_grads(grad_and_var)], 
        feed_dict = {input: train_input_data, lable: train_label_data})

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
