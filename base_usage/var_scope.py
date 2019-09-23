#%%
import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import lib.show_tb as tb
from datetime import datetime

#%%
# 1. variable_scope 跟 name_scope，是没有关系
# 2. variable_scope 使用了reuse，会搜索之前创建的
tf.reset_default_graph()
def dense(x, num_outputs, op_scope_name = None, var_scope_name=None):
    if var_scope_name == None:
        var_scope_name = "DenseVars"
    with tf.variable_scope(var_scope_name):
        weights = tf.get_variable("weights", shape=[int(x.shape[-1]), num_outputs],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[num_outputs], initializer=tf.constant_initializer(0.0))

    if op_scope_name == None:
        op_scope_name = "Dense"
    with tf.name_scope(op_scope_name):
        out = tf.nn.bias_add(tf.matmul(x, weights, name='matmul'), bias, name='bias_add')
    return out

x = tf.placeholder(dtype=tf.float32, shape=[None, 30], name="x")
xx = tf.placeholder(dtype=tf.float32, shape=[None, 30], name="xx")
with tf.name_scope("main") as main_scope:
    l1 = tf.nn.relu(dense(x, 10, var_scope_name="l1_vars", op_scope_name="l1"))
    y = tf.nn.relu(dense(l1, 5, var_scope_name="l2_vars", op_scope_name="l2"))

with tf.name_scope("test_sigmoid") as main_scope:
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        yy_t0l1= dense(xx, 10, var_scope_name="l1_vars", op_scope_name="layer1")
        yy_t0= dense(yy_t0l1, 5, var_scope_name="l2_vars", op_scope_name="layer2")
    yy_sigmoid = tf.nn.sigmoid(yy_t0)

with tf.name_scope("test_tanh") as main_scope:
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        yy_t1 = dense(xx, 10, op_scope_name="myDense", var_scope_name="l1_vars")
    yy_tanh = tf.nn.tanh(yy_t1)

tb.show_default_graph()



#%%
