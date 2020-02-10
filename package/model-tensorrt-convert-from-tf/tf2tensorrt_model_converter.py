#!/usr/bin/env python3

""" This is a standalone script for converting Onnx model files into TensorRT model files

    Author: Leo Gordon (dividiti)
"""


import argparse
import tensorrt as trt
import uff


def convert_tf_model_to_trt(tf_model_filename, trt_model_filename,
                               model_data_layout, input_layer_name, input_height, input_width,
                               output_layer_name, output_data_type, max_workspace_size, max_batch_size):
    "Convert an tf_model_filename into a trt_model_filename using the given parameters"

    uff_model = uff.from_tensorflow_frozen_model(tf_model_filename)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:

        if model_data_layout == 'NHWC':
            parser.register_input(input_layer_name, [input_height, input_width, 3], trt.UffInputOrder.NHWC)
        else:
            parser.register_input(input_layer_name, [3, input_height, input_width], trt.UffInputOrder.NCHW)

        parser.register_output(output_layer_name)

        if not parser.parse_buffer(uff_model, network):
            raise RuntimeError("UFF model parsing (originally from {}) failed. Error: {}".format(tf_model_filename, parser.get_error(0).desc()))

        if (output_data_type=='fp32'):
            print('Converting into fp32 (default), max_batch_size={}'.format(max_batch_size))
        else:
            if not builder.platform_has_fast_fp16:
                print('Warning: This platform is not optimized for fast fp16 mode')

            builder.fp16_mode = True
            print('Converting into fp16, max_batch_size={}'.format(max_batch_size))

        builder.max_workspace_size  = max_workspace_size
        builder.max_batch_size      = max_batch_size


        trt_model_object    = builder.build_cuda_engine(network)

        try:
            serialized_trt_model = trt_model_object.serialize()
            with open(trt_model_filename, "wb") as trt_model_file:
                trt_model_file.write(serialized_trt_model)
        except:
            raise RuntimeError('Cannot serialize or write TensorRT engine to file {}.'.format(trt_model_filename))


def main():
    "Parse command line and feed the conversion function"

    arg_parser  = argparse.ArgumentParser()
    arg_parser.add_argument('tf_model_filename',    type=str,                       help='TensorFlow model file')
    arg_parser.add_argument('trt_model_filename',   type=str,                       help='TensorRT model file')
    arg_parser.add_argument('--model_data_layout',  type=str,   default='NHWC',     help='Model data layout (NHWC or NCHW)')
    arg_parser.add_argument('--input_layer_name',   type=str,   default='input',    help='Input layer name')
    arg_parser.add_argument('--input_height',       type=int,   default=224,        help='Input height')
    arg_parser.add_argument('--input_width',        type=int,   default=224,        help='Input width')
    arg_parser.add_argument('--output_layer_name',  type=str,   default='MobilenetV1/Predictions/Reshape_1', help='Output layer name')
    arg_parser.add_argument('--output_data_type',   type=str,   default='fp32',     help='Model data type')
    arg_parser.add_argument('--max_workspace_size', type=int,   default=(1<<30),    help='Builder workspace size')
    arg_parser.add_argument('--max_batch_size',     type=int,   default=1,          help='Builder batch size')
    args        = arg_parser.parse_args()

    convert_tf_model_to_trt( args.tf_model_filename, args.trt_model_filename,
                                args.model_data_layout, args.input_layer_name, args.input_height, args.input_width,
                                args.output_layer_name, args.output_data_type, args.max_workspace_size, args.max_batch_size )

main()

