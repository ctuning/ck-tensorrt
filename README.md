# Collective Knowledge repository for collaboratively benchmarking and optimising embedded deep vision runtime library for Jetson TX1

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction

[CK-TensorRT](https://github.com/ctuning/ck-tensorrt) is an open framework for
collaborative and reproducible optimisation of convolutional neural networks for Jetson TX1.
It's based on the [Deep Inference](https://github.com/dusty-nv/jetson-inference) framework from
Dustin Franklin (a [Jetson developer @ NVIDIA](https://github.com/dusty-nv)) and
the [Collective Knowledge](http://cknowledge.org) framework for customizable
cross-platform experimental workflows
 from the [cTuning Foundation](http://ctuning.org). In essence, CK-TensorRT 
is simply a suite of convenient wrappers with unified JSON API 
for customizable building, evaluating and multi-objective optimisation 
of Jetson Inference runtime library for Jetson TX1.

See [project page](http://cKnowledge.org/ai) for more details.

![](http://cKnowledge.org/images/ai-cloud-resize.png)

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
using Collective Knowledge Technology:
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-Caffe2](https://github.com/ctuning/ck-caffe2)
* [Reusable AI artifacts](http://cKnowledge.org/ai-artifacts)
* [CK-TinyDNN](https://github.com/ctuning/ck-tiny-dnn)
* [Android app for DNN crowd-benchmarking and crowd-tuning](http://cKnowledge.org/android-apps.html)
* [CK-powered ARM workload automation](https://github.com/ctuning/ck-wa)

## Open R&D challenges

We use crowd-benchmarking and crowd-tuning of such realistic workloads across diverse hardware for 
[open academic and industrial R&D challenges](https://github.com/ctuning/ck/wiki/Research-and-development-challenges.mediawiki) - 
join this community effort!

## Related publications with long term vision

* <a href="https://github.com/ctuning/ck/wiki/Publications">All references with BibTeX</a>
