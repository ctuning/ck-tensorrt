#!/usr/bin/env python3

""" This is a standalone script for converting Onnx model files into TensorRT model files

    Author: Leo Gordon (dividiti)
"""


import argparse
import tensorrt as trt


def convert_onnx_model_to_trt(onnx_model_filename, trt_model_filename,
                              input_tensor_name, output_tensor_name,
                              output_data_type, max_workspace_size, max_batch_size):
    "Convert an onnx_model_filename into a trt_model_filename using the given parameters"

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    TRT_VERSION_MAJOR = int(trt.__version__.split('.')[0])

    with trt.Builder(TRT_LOGGER) as builder:
        if TRT_VERSION_MAJOR >= 7:
            flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)) | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            network = builder.create_network(flag)
        else:
            network = builder.create_network()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if (output_data_type=='fp32'):
            print('Converting into fp32 (default), max_batch_size={}'.format(max_batch_size))
            builder.fp16_mode = False
        else:
            if not builder.platform_has_fast_fp16:
                print('Warning: This platform is not optimized for fast fp16 mode')

            builder.fp16_mode = True
            print('Converting into fp16, max_batch_size={}'.format(max_batch_size))

        builder.max_workspace_size  = max_workspace_size
        builder.max_batch_size      = max_batch_size

        with open(onnx_model_filename, 'rb') as onnx_model_file:
            onnx_model = onnx_model_file.read()

        if not parser.parse(onnx_model):
            raise RuntimeError("Onnx model parsing from {} failed. Error: {}".format(onnx_model_filename, parser.get_error(0).desc()))

        if TRT_VERSION_MAJOR >= 7:
            # Create an optimization profile (see Section 7.2 of https://docs.nvidia.com/deeplearning/sdk/pdf/TensorRT-Developer-Guide.pdf).
            profile = builder.create_optimization_profile()
            # FIXME: Hardcoded for ImageNet. The minimum/optimum/maximum dimensions of a dynamic input tensor are the same.
            profile.set_shape(input_tensor_name, (1, 3, 224, 224), (max_batch_size, 3, 224, 224), (max_batch_size, 3, 224, 224))

            config = builder.create_builder_config()
            config.add_optimization_profile(profile)

            trt_model_object = builder.build_engine(network, config)
        else:
            trt_model_object = builder.build_cuda_engine(network)

        try:
            serialized_trt_model = trt_model_object.serialize()
            with open(trt_model_filename, "wb") as trt_model_file:
                trt_model_file.write(serialized_trt_model)
        except:
            raise RuntimeError('Cannot serialize or write TensorRT engine to file {}.'.format(trt_model_filename))


def main():
    "Parse command line and feed the conversion function"

    arg_parser  = argparse.ArgumentParser()
    arg_parser.add_argument('onnx_model_file',      type=str,                             help='Onnx model file')
    arg_parser.add_argument('trt_model_filename',   type=str,                             help='TensorRT model file')
    arg_parser.add_argument('--input_tensor_name',  type=str,   default='input_tensor:0', help='Input tensor type')
    arg_parser.add_argument('--output_tensor_name', type=str,   default='prob',           help='Output tensor type')
    arg_parser.add_argument('--output_data_type',   type=str,   default='fp32',           help='Model data type')
    arg_parser.add_argument('--max_workspace_size', type=int,   default=(1<<30),          help='Builder workspace size')
    arg_parser.add_argument('--max_batch_size',     type=int,   default=1,                help='Builder batch size')
    args        = arg_parser.parse_args()

    convert_onnx_model_to_trt( args.onnx_model_file, args.trt_model_filename,
                               args.input_tensor_name, args.output_tensor_name,
                               args.output_data_type, args.max_workspace_size, args.max_batch_size )

main()

