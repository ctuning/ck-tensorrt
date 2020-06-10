#!/usr/bin/env python3

import os
import numpy as np
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_PLUGIN_PATH       = os.getenv('CK_ENV_TENSORRT_PLUGIN_PATH', os.getenv('ML_MODEL_TENSORRT_PLUGIN',''))
MODEL_USE_DLA           = os.getenv('ML_MODEL_USE_DLA', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))

## Processing in batches:
#
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))


if MODEL_PLUGIN_PATH:
    import ctypes
    if not os.path.isfile(MODEL_PLUGIN_PATH):
        raise IOError("{}\n{}\n".format(
            "Failed to load library ({}).".format(MODEL_PLUGIN_PATH),
            "Please build the plugin."
        ))
    ctypes.CDLL(MODEL_PLUGIN_PATH)


def initialize_predictor():
    global pycuda_context
    global d_inputs, h_d_outputs, h_output, model_bindings, cuda_stream
    global input_volume, output_volume
    global trt_context
    global BATCH_SIZE
    global max_batch_size
    global trt_version

    # Load the TensorRT model from file
    pycuda_context = pycuda.tools.make_default_context()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    try:
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(MODEL_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            trt_version = [ int(v) for v in trt.__version__.split('.') ]
            print('[TensorRT v{}.{}] successfully loaded'.format(trt_version[0], trt_version[1]))
    except:
        pycuda_context.pop()
        raise RuntimeError('TensorRT model file {} is not found or corrupted'.format(MODEL_PATH))

    max_batch_size      = trt_engine.max_batch_size

    if trt_version[0] >= 7 and BATCH_SIZE>1:
        pycuda_context.pop()
        raise RuntimeError("Desired batch_size ({}) is not yet supported in TensorRT {}".format(BATCH_SIZE,trt_version[0]))

    if BATCH_SIZE>max_batch_size:
        pycuda_context.pop()
        raise RuntimeError("Desired batch_size ({}) exceeds max_batch_size of the model ({})".format(BATCH_SIZE,max_batch_size))

    trt_context     = trt_engine.create_execution_context()

    d_inputs, h_d_outputs, model_bindings = [], [], []
    for interface_layer in trt_engine:
        idx     = trt_engine.get_binding_index(interface_layer)
        dtype   = trt_engine.get_binding_dtype(interface_layer)
        shape   = tuple(abs(i) for i in trt_engine.get_binding_shape(interface_layer))
        fmt     = trt_engine.get_binding_format(idx) if trt_version[0] >= 6 else None

        if fmt and fmt == trt.TensorFormat.CHW4 and trt_engine.binding_is_input(interface_layer):
            shape[-3] = ((shape[-3] - 1) // 4 + 1) * 4
        size    = trt.volume(shape) * max_batch_size

        dev_mem = cuda.mem_alloc(size * dtype.itemsize)
        model_bindings.append( int(dev_mem) )

        if trt_engine.binding_is_input(interface_layer):
            if trt_version[0] >= 6:
                trt_context.set_binding_shape(idx, shape)
            interface_type = 'Input'
            d_inputs.append(dev_mem)
            model_input_shape   = shape
        else:
            interface_type = 'Output'
            host_mem    = cuda.pagelocked_empty(size, trt.nptype(dtype))
            h_d_outputs.append({ 'host_mem': host_mem, 'dev_mem': dev_mem })
            if MODEL_SOFTMAX_LAYER=='' or interface_layer == MODEL_SOFTMAX_LAYER:
                model_output_shape  = shape
                h_output            = host_mem

        print("{} layer {}: dtype={}, shape={}, elements_per_max_batch={}".format(interface_type, interface_layer, dtype, shape, size))

    cuda_stream     = cuda.Stream()
    input_volume    = trt.volume(model_input_shape)     # total number of monochromatic subpixels (before batching)
    output_volume   = trt.volume(model_output_shape)    # total number of elements in one image prediction (before batching)
    num_layers      = trt_engine.num_layers

    return pycuda_context, max_batch_size, input_volume, output_volume, num_layers


def inference_for_given_batch(batch_data):
    global d_inputs, h_d_outputs, h_output, model_bindings, cuda_stream
    global trt_context
    global max_batch_size
    global trt_version

    actual_batch_size  = len(batch_data)
    if MODEL_USE_DLA and max_batch_size>actual_batch_size:
        batch_data = np.pad(batch_data, ((0,max_batch_size-actual_batch_size), (0,0), (0,0), (0,0)), 'constant')
        pseudo_batch_size   = max_batch_size
    else:
        pseudo_batch_size   = actual_batch_size

    flat_batch  = np.ravel(batch_data)

    begin_inference_timestamp   = time.time()

    cuda.memcpy_htod_async(d_inputs[0], flat_batch, cuda_stream)  # assuming one input layer for image classification
    if trt_version[0] >= 7:
        trt_context.execute_async_v2(bindings=model_bindings, stream_handle=cuda_stream.handle)
    else:
        trt_context.execute_async(bindings=model_bindings, batch_size=pseudo_batch_size, stream_handle=cuda_stream.handle)

    for output in h_d_outputs:
        cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)
    cuda_stream.synchronize()

    inference_time_s   = time.time() - begin_inference_timestamp

    ## first dimension contains actual_batch_size vectors, further format depends on the task:
    #
    trimmed_batch_results   = np.split(h_output, max_batch_size)[:actual_batch_size]

    return trimmed_batch_results, inference_time_s

