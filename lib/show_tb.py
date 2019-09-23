""" Code taken from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
Placed here for convenience only.
"""
from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf
import numpy as np

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                #tensor.tensor_content = "<stripped 1 bytes>"
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, 'utf-8')
    return strip_def

def save_default_graph(dir="logs"):
  from datetime import datetime
  import os
  TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
  tf.summary.FileWriter(os.path.join(dir,  TIMESTAMP), tf.get_default_graph())

def show_default_graph(max_const_size=32):
  show_graph(tf.get_default_graph(), max_const_size)

def show_graph(graph_or_graphdef, max_const_size=32):
    """Visualize TensorFlow graph."""
    graph_def = graph_or_graphdef
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

