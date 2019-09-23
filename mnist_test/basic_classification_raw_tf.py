#%%
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from lib.show_tb import show_default_graph

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#%%
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#%%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#%%
train_images = train_images / 255.0
test_images = test_images / 255.0

#%%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#%%
class Model:
  def __init__(self):
    tf.reset_default_graph()
    input = tf.placeholder(tf.float32, [None, 28, 28], name='input')
    labels = tf.placeholder(tf.int32, [None], name='label')
    
    f1 = tf.contrib.layers.flatten(input)

    w2 = tf.Variable(tf.random_normal([28*28, 128], dtype=tf.float32))
    b2 = tf.Variable(tf.random_normal([1, 128], dtype=tf.float32))
    f2 = tf.nn.relu(tf.matmul(f1, w2) + b2)

    w3 = tf.Variable(tf.random_normal([128, 10], dtype=tf.float32))
    b3 = tf.Variable(tf.random_normal([1, 10], dtype=tf.float32))
    f3 =tf.matmul(f2, w3) + b3
    self.weights = [w2, w3]
    self.bias = [b2, b3]
    self.f3 = f3
    self.input = input
    self.labels = labels

    #因为labes最后一维是1， 是个数值，不是向量，所以这里需要使用sparse
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = self.f3))
    self.prediction =  tf.nn.softmax(self.f3)
    #self.accuracy  = tf.metrics.accuracy(labels = self.labels, predictions=tf.math.argmax(self.prediction))

    opt =  tf.train.AdamOptimizer(0.01)
    grads_and_vars = opt.compute_gradients(self.loss)
    self.train_step = opt.apply_gradients(grads_and_vars)
    init  = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)
    self.grads_and_vars = grads_and_vars
    self.all_grads = []

    #compute acc
    prediction_index = tf.argmax(self.prediction, axis=-1, output_type=tf.int32)
    equality = tf.math.equal(labels, prediction_index)
    self.acc =  tf.reduce_mean(tf.cast(equality, tf.float32)) 

  def __del__(self):
    self.sess.close()

  def train(self, train_images, train_labels, epochs=5):
    batch_size = 1000
    batch_step = len(train_images) // batch_size
  
    for _ in range(epochs):
      for b in range(batch_step):
        self.sess.run(self.train_step, feed_dict = {self.input: train_images[b * batch_size :(b + 1)*batch_size],
          self.labels: train_labels[b * batch_size :(b + 1)*batch_size]} )
      
      print("loss:", self.sess.run(self.loss, feed_dict =  {self.input: train_images,
          self.labels: train_labels}))
      grads = self.sess.run(self.grads_and_vars, feed_dict =  {self.input: train_images,
          self.labels: train_labels})
      #print("grads:", grads)

      self.all_grads.append(grads)
      accuracy = self.sess.run(self.acc, feed_dict =  {self.input: train_images,
          self.labels: train_labels})
      print("my accuracy:", accuracy) 


  def accuracy(self, input_images, labels):
    accuracy = self.sess.run(self.acc, feed_dict =  {self.input: input_images,
          self.labels: labels})
    return accuracy

  def predict(self, input_images):
    return self.sess.run(self.prediction, feed_dict = {self.input: input_images} )

model = Model()
model.train(train_images, train_labels, epochs=50)
predictions = model.predict(test_images)

print("==============================")
print("test data accuracy:", model.accuracy(test_images, test_labels))



#%%
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#%%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#%%

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.show()

#%%
#show_graph(K.get_session().graph)
show_default_graph()


#%%
