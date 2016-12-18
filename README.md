# Collective Knowledge repository for collaboratively benchmarking and optimising embedded deep vision runtime library for Jetson TX1

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction

[CK-TensorRT](https://github.com/dividiti/ck-tensorrt) is an open framework for
collaborative and reproducible optimisation of convolutional neural networks for Jetson TX1.
It's based on the [Deep Inference](https://github.com/dusty-nv/jetson-inference) framework from the
Dustin Franklin ([Jetson Developer @NVIDIA](https://github.com/dusty-nv)) and
the [Collective Knowledge](http://cknowledge.org) framework from the [cTuning
Foundation](http://ctuning.org). In essence, CK-TensorRT is simply a suite of
convenient wrappers for building, evaluating and optimising performance of
Jetson Inference runtime library for Jetson TX1.

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
TBD

### Installing CK

```
$ sudo pip install ck
$ ck version
```

### Installing CK-TensorRT repository

```
$ ck pull repo:ck-tensorrt --url=https://github.com/dividiti/ck-tensorrt
```

### Building CK-TensorRT and all dependencies via CK

The first time you run the TensorRT benchmark, CK will
build and install all missing dependencies on your machine,
download the required data sets and start the benchmark:

```
$ ck run program:imagenet-console
```

## Related projects and initiatives

We are working with the community to unify and crowdsource performance analysis 
and tuning of various DNN frameworks (or any realistic workload) 
using Collective Knowledge Technology:
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-TinyDNN](https://github.com/ctuning/ck-tiny-dnn)
* [Android app for DNN crowd-benchmarking and crowd-tuning](https://play.google.com/store/apps/details?id=openscience.crowdsource.video.experiments)
* [CK-powered ARM workload automation](https://github.com/ctuning/ck-wa)

## Open R&D challenges

We use crowd-benchmarking and crowd-tuning of such realistic workloads across diverse hardware for 
[open academic and industrial R&D challenges](https://github.com/ctuning/ck/wiki/Research-and-development-challenges.mediawiki) - 
join this community effort!

## Related publications with long term vision

* <a href="https://github.com/ctuning/ck/wiki/Publications">All references with BibTex</a>
