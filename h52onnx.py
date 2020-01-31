import os
# Set env variable
os.environ['TF_KERAS'] = '1'

import keras2onnx
import tensorflow as tf
import onnx

path_to_model = '/Users/fei/Documents/sandbox/python/donkey/donkeycar_app/models/rnn_imu_2.h5'

model = tf.keras.models.load_model(path_to_model)
onnx_model = keras2onnx.convert_keras(model)
onnx.save_model(onnx_model, 'rnn_imu_2.onnx')