# tf_model_convert

## Env
python: 3.7
tf: 1.15.3
os: ubuntu 20.04 x64

## Cmd

./converter_lite --fmk=TFLITE --modelFile=./matmul.tflite --outputFile=matmul

./flatc --json --strict-json --defaults-json --unknown-json ./model.fbs --raw-binary -- matmul.ms

./benchmark --accuracyThreshold=0.05 --inDataPath=./matmul.bin --loopCount=1 --modelPath=./matmul.ms --calibDataPath=./matmul.out --numThreads=1 --fp16Priority=true