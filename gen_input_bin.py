import sys
import numpy as np

def main(argv):
  data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
  data.tofile("matmul.bin")

if __name__ == "__main__":
  main(sys.argv)