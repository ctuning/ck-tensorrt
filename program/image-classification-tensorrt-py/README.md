# Image Classification - TensorRT-Python program

The instructions below have been tested on a Jetson TX1 board with JetPack 4.2.2 installed via the NVIDIA SDK Manager.

## Convert TF model to ONNX model

When installing a Jetpack via the NVIDIA SDK Manager, tick the TensorFlow option.
For JetPack 4.2.2, this installs TensorFlow 1.14.0.

### Detect TensorFlow
```
$ ck detect soft:lib.tensorflow --full_path=/usr/local/lib/python3.6/dist-packages/tensorflow/__init__.py
```

### Install ONNX from source (with the ProtoBuf compiler dependency)
```
$ ck install package --tags=lib,python-package,onnx,from-source
```

### Install TF-to-ONNX converter (of a known good version)
```
$ ck install package --tags=lib,python-package,tf2onnx --force_version=1.5.1
```
**NB:** Both 1.5.2. and 1.5.3 can be installed but fail to convert ResNet to ONNX on TX1.

### Convert TF to ONNX
```
$ ck install package --tags=model,resnet,onnx,converted-from-tf
```

### Convert ONNX to TensorRT

When converting an ONNX model to TensorRT, you can select the numerical data type (`fp32` or `fp16`)
and the maximum batch size (currently `1 .. 20`).

#### `precision=fp32`, `max_batch_size=1`
```
$ ck install package --tags=model,resnet,tensorrt,converted-from-onnx
```

#### `precision=fp16`, `max_batch_size=1`
```
$ ck install package --tags=model,resnet,tensorrt,converted-from-onnx,fp16
```

#### `precision=fp32`, `max_batch_size=2`
```
$ ck install package --tags=model,resnet,tensorrt,converted-from-onnx,fp32,maxbatch.2
```

#### `precision=fp16`, `max_batch_size=2`
```
$ ck install package --tags=model,resnet,tensorrt,converted-from-onnx,fp16,maxbatch.2
```
