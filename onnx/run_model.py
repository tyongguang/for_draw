#%%
import onnx
import caffe2.python.onnx.backend as cf_backend
import numpy as np
import onnxruntime as rt
import onnx_tf.backend as tf_backend
import tensorflow as tf 
import nnvm
import tvm

#%%
input_data = np.array([[
    -0.41774908, -0.4086338 , -0.39947757, -0.39028135, -0.381046 ,-0.37177247, -0.3624617 , -0.3531146 , -0.34373212, -0.3343152 
]], dtype=np.float32)
input_data = input_data.reshape(-1, 1, 10)
expect = np.array([-0.3248648])


#%%
model = onnx.load("lstm_sin.onnx")  # load onnx model
sym = nnvm.frontend.from_onnx(model)



#%%
# 使用tensorflow 作为后端
# 低级错误一直不改，"input:0"不能作为placeholder名字
model = onnx.load("lstm_sin.onnx")  # load onnx model
tf_rep = tf_backend.prepare(model)
tf_output = tf_rep.run(input_data)
print(tf_output)



#%%
# 使用CAFFE的后端，暂时不work
# 算子不支持
model = onnx.load("lstm_sin.onnx")
onnx.checker.check_model(model)
cf_rep = cf_backend.prepare(model, device="CPU") # or "CPU"
cf_output = cf_rep.run(input_data)
print(cf_output)

#%%
# 使用微软的后端
# OK
sess = rt.InferenceSession("lstm_sin.onnx")
input_name = sess.get_inputs()[0].name
ms_output = sess.run(None, {input_name: input_data})
print(ms_output)



#%%
