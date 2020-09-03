import tensorflow as tf
import numpy as np
import sys

def get_tensor_data(interpreter, name):
  tensors = interpreter.get_tensor_details()
  for tensor in tensors:
    if tensor['name'] == name:
      return interpreter.get_tensor(tensor['index'])
  raise Exception('Not find tensor: ' + name)


def main(argv):
  input_file = 'matmul.bin'
  model_file = 'matmul.tflite'
  output_file = 'matmul.out'
  output_name = 'output'
  data = np.fromfile(input_file, dtype=np.float32)
  data = data.reshape(4, 2)
  interpreter = tf.lite.Interpreter(model_file)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]['index'], data)
  interpreter.invoke()
  
  output = get_tensor_data(interpreter, output_name)
  tile = 'FullyConnected-0' + ' ' + str(len(output.shape)) + ' '
  with open(output_file, 'w') as txt:
    for i in range(len(output.shape)):
      tile += str(output.shape[i]) + ' '
    txt.write(tile + '\n')

    op_flatten = np.squeeze(output).flatten()
    size = op_flatten.shape[0]
    for k in range(size):
      txt.write(str(op_flatten[k]) + '\n')
    txt.write('\n')

if __name__ == "__main__":
  main(sys.argv)