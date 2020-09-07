# tf_model_convert

## Env
python: 3.7
tf: 1.15.3
os: ubuntu 20.04 x64

## Cmd

./converter_lite --fmk=TFLITE --modelFile=./matmul.tflite --outputFile=matmul

./flatc --json --strict-json --defaults-json --unknown-json ./model.fbs --raw-binary -- matmul.ms

run gen_input_bin.py to generate input bin file. In matmul.json, find the output node name in ms model, and write it to line 27 of inference.py. Run inference.py to generate matmul.out.

./benchmark --accuracyThreshold=0.05 --inDataPath=./matmul.bin --loopCount=1 --modelPath=./matmul.ms --calibDataPath=./matmul.out --numThreads=1 --fp16Priority=true

mindspore@
1
2
3