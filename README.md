# Collective Knowledge repository for collaboratively benchmarking and optimising embedded deep vision runtime library for Jetson TX1

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction

[CK-Tensorrt](https://github.com/dividiti/ck-tensorrt) is an open framework for
collaborative and reproducible optimisation of convolutional neural networks for Jetson TX1.
It's based on the [Deep Inference](https://github.com/dusty-nv/jetson-inference) framework from the
Dustin Franklin ([Jetson Developer @NVIDIA](https://github.com/dusty-nv)) and
the [Collective Knowledge](http://cknowledge.org) framework from the [cTuning
Foundation](http://ctuning.org). In essence, CK-Tensorrt is simply a suite of
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

### Installing CK-Tensorrt dependencies
TBD

### Installing CK

```
$ sudo pip install ck
$ ck version
```

### Installing CK-Tensorrt repository

```
$ ck pull repo:ck-tensorrt --url=https://github.com/dividiti/ck-tensorrt
```

### Building CK-Tensorrt and all dependencies via CK

The first time you run Tensorrt benchmark, CK will
build and install all missing dependencies for your machine,
download required data sets and will start benchmark:

```
$ ck run program:imagenet-console
```

## Preliminary results

### Compare accuracy of 4 CNNs on Jetson TX1

TBD

we compare the Top-1 and Top-5 accuracy of 4 CNNs:

- [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
- [SqueezeNet 1.0](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0)
- [SqueezeNet 1.1](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1)
- [GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)

on the [Imagenet validation set](http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) (50,000 images).

We have thus independently verified that on this data set [SqueezeNet](https://arxiv.org/abs/1602.07360) matches (and even slightly exceeds) the accuracy of [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

The experimental data is stored in the main CK-Caffe repository under '[experiment](https://github.com/dividiti/ck-caffe/tree/master/experiment)'.

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

## Related Publications with long term vision

* <a href="https://github.com/ctuning/ck/wiki/Publications">All references with BibTex</a>
