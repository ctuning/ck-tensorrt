# Collective Knowledge repository for collaboratively benchmarking and optimising embedded deep vision runtime library for Jetson TX1

**All CK components can be found at [cKnowledge.io](https://cKnowledge.io) and in [one GitHub repository](https://github.com/ctuning/ck-mlops)!**

*This project is hosted by the [cTuning foundation](https://cTuning.org).*


[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction

[CK-TensorRT](https://github.com/ctuning/ck-tensorrt) is an open framework for
collaborative and reproducible optimisation of convolutional neural networks for Jetson TX1
based on the [Collective Knowledge](http://cknowledge.org) framework. 
It's based on the [Deep Inference](https://github.com/dusty-nv/jetson-inference) framework from
Dustin Franklin (a [Jetson developer @ NVIDIA](https://github.com/dusty-nv)).
In essence, CK-TensorRT is simply a suite of convenient wrappers with unified JSON API 
for customizable building, evaluating and multi-objective optimisation 
of Jetson Inference runtime library for Jetson TX1.

## Authors/contributors

* Anton Lokhmotov, [dividiti](http://dividiti.com)
* Daniil Efremov, [xored](http://xored.com)

## Quick installation on Ubuntu

TBD

### Installing general dependencies

```
$ sudo apt install coreutils \
                   build-essential \
                   make \
                   cmake \
                   wget \
                   git \
                   python \
                   python-pip
```

### Installing CK-TensorRT dependencies
```
$ sudo apt install libqt4-dev \
                   libglew-dev \
                   libgstreamer1.0-dev
```

### Installing CK

```
$ sudo pip install ck
$ ck version
```

### Installing CK-TensorRT repository

```
$ ck pull repo:ck-tensorrt
```

### Building CK-TensorRT and all dependencies via CK

The first time you run a TensorRT program (e.g. `tensorrt-test`), CK will
build and install all missing dependencies on your machine,
download the required data sets and start the benchmark:

```
$ ck run program:tensorrt-test
```

## Related projects and initiatives

We are working with the community to unify and crowdsource performance analysis 
and tuning of various DNN frameworks (or any realistic workload) 
using the Collective Knowledge Technology:
* [Open repository of AI, ML, and systems knowledge](https://cKnowledge.io)
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-Caffe2](https://github.com/ctuning/ck-caffe2)
* [Android app for DNN crowd-benchmarking and crowd-tuning]( https://cKnowledge.org/android-apps.html )
