import tensorflow as tf
import numpy as np

tf_model_path = './saved_model/'
tflite_model_path = 'resnet34_peoplenet_int8.tflite'

# Generate representative dataset
def representative_dataset():
    data = tf.random.uniform((1,544,960,3))
    yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# converter.representative_dataset = representative_dataset()
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Can be tf.uint8, or tf.float32 or tf.float16
converter.inference_output_type = tf.float32  # Can be tf.uint8, tf.int8 or tf.float16. We keep it float32 for ease of post-processing output data

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)