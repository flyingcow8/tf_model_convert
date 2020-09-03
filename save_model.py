import tensorflow as tf
from tensorflow.python.framework import graph_util


def lite_converter(pb_file, tflite_file, input_name, output_name):
    input_arrays = [input_name]
    output_arrays = [output_name]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_file, input_arrays, output_arrays)
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open(tflite_file, 'wb').write(tflite_model)


pb_file = 'matmul.pb'
tf_input = tf.placeholder(name='input', dtype='float32', shape=[4, 2])
weight = tf.constant([1, 2, 3, 4, 5, 6], dtype='float32', shape=[2, 3])
output = tf.matmul(tf_input, weight, name='output')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['output'])
    with tf.gfile.FastGFile(pb_file, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

tflite_file = 'matmul.tflite'
lite_converter(pb_file, tflite_file, 'input', 'output')
