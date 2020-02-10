#!/usr/bin/env python3

""" This is a standalone script for converting Caffe model files into TensorRT model files

    Author: Leo Gordon (dividiti)
"""


import argparse
import tensorrt as trt


def convert_caffe_model_to_trt(caffe_weights_file, caffe_deploy_file, trt_model_filename,
                               output_tensor_name, output_data_type, max_workspace_size, max_batch_size):
    "Convert a pair of (caffe_weights_file,caffe_deploy_file) into a trt_model_file using the given parameters"

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:

        if (output_data_type=='fp16'):
            if not builder.platform_has_fast_fp16:
                print('Warning: This platform is not optimized for fast fp16 mode')

            builder.fp16_mode = True
            print('Converting into fp16, max_batch_size={}'.format(max_batch_size))
        else:
            print('Converting into fp32 (default), max_batch_size={}'.format(max_batch_size))

        builder.max_workspace_size  = max_workspace_size
        builder.max_batch_size      = max_batch_size

        model_tensors       = parser.parse(deploy=caffe_deploy_file, model=caffe_weights_file, network=network, dtype=trt.float32)
        network.mark_output(model_tensors.find(output_tensor_name))

        trt_model_object    = builder.build_cuda_engine(network)

        try:
            serialized_trt_model = trt_model_object.serialize()
            with open(trt_model_filename, "wb") as trt_model_file:
                trt_model_file.write(serialized_trt_model)
        except:
            print('Error: cannot serialize or write TensorRT engine to file {}.'.format(trt_model_filename))

def main():
    "Parse command line and feed the conversion function"

    arg_parser  = argparse.ArgumentParser()
    arg_parser.add_argument('caffe_weights_file',   type=str,                       help='Caffe model weights file')
    arg_parser.add_argument('caffe_deploy_file',    type=str,                       help='Caffe model deploy file')
    arg_parser.add_argument('trt_model_filename',   type=str,                       help='TensorRT model file')
    arg_parser.add_argument('--output_tensor_name', type=str,   default='prob',     help='Output tensor type')
    arg_parser.add_argument('--output_data_type',   type=str,   default='fp32',     help='Model data type')
    arg_parser.add_argument('--max_workspace_size', type=int,   default=(1<<30),    help='Builder workspace size')
    arg_parser.add_argument('--max_batch_size',     type=int,   default=1,          help='Builder batch size')
    args        = arg_parser.parse_args()

    convert_caffe_model_to_trt( args.caffe_weights_file, args.caffe_deploy_file, args.trt_model_filename,
                                args.output_tensor_name, args.output_data_type, args.max_workspace_size, args.max_batch_size )

main()

