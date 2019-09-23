import tensorflow as tf
import os


def accuracy(softmax_output, one_of_n_labels):
    prediction_index = tf.argmax(softmax_output, axis=-1)
    labels_index =  tf.argmax(one_of_n_labels, axis=-1)
    equality = tf.math.equal(labels_index, prediction_index)
    acc =  tf.reduce_mean(tf.cast(equality, tf.float32))
    return acc

def accuracy_sparse(softmax_output, sparse_labels):
    prediction_index = tf.argmax(softmax_output, axis=-1)
    equality = tf.math.equal(sparse_labels, prediction_index)
    acc =  tf.reduce_mean(tf.cast(equality, tf.float32))
    return acc


def get_vars(grad_and_vars):
    return [g_v[1] for g_v in grad_and_vars ]

def get_grads(grad_and_vars):
    return [g_v[0] for g_v in grad_and_vars ]

def get_names(grad_and_vars):
    return [g_v[1].name for g_v in grad_and_vars ]

def save_as_pb(file_path, sess, input_node_names, output_node_names, as_text = False):
    from google.protobuf import text_format
    from tensorflow.python.framework import graph_util
    from tensorflow.tools.graph_transforms import TransformGraph

    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_convert_variables_to_constants(sess, 
        graph_def, output_node_names)
   
    transforms = ["fold_constants(ignore_errors=true)",
                "fold_batch_norms",
                "fold_old_batch_norms"
                ]
    
    output_graph_def = TransformGraph(output_graph_def, input_node_names,
                                      output_node_names, transforms)

    with tf.gfile.GFile(file_path, "wb") as f:
        if as_text:
            f.write(text_format.MessageToString(output_graph_def))
        else:
            f.write(output_graph_def.SerializeToString())
    return output_graph_def

def import_pb(file_path):
    with tf.gfile.FastGFile(file_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


class TrainSaver:
    def __init__(self, path, scope=None, max_to_keep=3):
        """
        path: 一部分路径，一部分名字。如：my_dir/my_model.ckpt
        """
        if scope == None:
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        else:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.saver = tf.train.Saver(var_list)     
        self.path = path
        dir_name = os.path.dirname(self.path)
        if False == os.path.exists(dir_name):
            os.mkdir(dir_name)

    def try_load(self, sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        return False

    def remove_old_ckpt(self, sess, global_step):
        parent_dir = os.path.dirname(self.path)
        files = os.listdir(parent_dir)
        for f in files:
            os.remove(os.path.join(parent_dir, f))
        self.saver.save(sess, self.path, global_step=global_step)

    def save(self, sess, global_step):
        self.saver.save(sess, self.path, global_step=global_step)

def hist_grad_vars(saver, grad_and_var, feed_dict):
    import matplotlib.pyplot as plt 
    with tf.Session() as sess:
        if False == saver.try_load(sess):
            raise Exception("not train?")

        vars = sess.run(get_vars(grad_and_var))
        grads = sess.run(get_grads(grad_and_var), feed_dict=feed_dict)

        names = get_names(grad_and_var)

        for v, name in zip(vars, names):
            plt.title("var:[%s]" %  name)
            plt.hist(v.flatten(), bins=50)
            plt.show()
        print("----------------------")
        for g, name in zip(grads, names):
            plt.title("grad:[%s]" % name)
            plt.hist(g.flatten(), bins=50)
            plt.show()
    