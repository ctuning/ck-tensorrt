#!/usr/bin/env python3

import json
import time
import os
import shutil
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools


## Post-detection filtering by confidence score:
#
SCORE_THRESHOLD = float(os.getenv('CK_DETECTION_THRESHOLD', 0.0))


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_PLUGIN_PATH       = os.getenv('ML_MODEL_TENSORRT_PLUGIN','')
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
LABELS_PATH             = os.environ['ML_MODEL_CLASS_LABELS']
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_MAX_PREDICTIONS   = int(os.getenv('ML_MODEL_MAX_PREDICTIONS', 100))
MODEL_SKIPPED_CLASSES   = os.getenv("ML_MODEL_SKIPS_ORIGINAL_DATASET_CLASSES", None)

if (MODEL_SKIPPED_CLASSES):
    SKIPPED_CLASSES = [int(x) for x in MODEL_SKIPPED_CLASSES.split(",")]
else:
    SKIPPED_CLASSES = None


if MODEL_PLUGIN_PATH:
    import ctypes
    if not os.path.isfile(MODEL_PLUGIN_PATH):
        raise IOError("{}\n{}\n".format(
            "Failed to load library ({}).".format(MODEL_PLUGIN_PATH),
            "Please build the plugin."
        ))
    ctypes.CDLL(MODEL_PLUGIN_PATH)


## Internal processing:
#
VECTOR_DATA_TYPE        = np.float32

## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.array(GIVEN_CHANNEL_MEANS.split(' '), dtype=VECTOR_DATA_TYPE)
    if MODEL_COLOURS_BGR:
        GIVEN_CHANNEL_MEANS = GIVEN_CHANNEL_MEANS[::-1]     # swapping Red and Blue colour channels

## Input image properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR')
IMAGE_LIST_FILE_NAME    = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF')
IMAGE_LIST_FILE         = os.path.join(IMAGE_DIR, IMAGE_LIST_FILE_NAME)
IMAGE_DATA_TYPE         = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DATA_TYPE', 'uint8')

## Writing the results out:
#
CUR_DIR = os.getcwd()
DETECTIONS_OUT_DIR      = os.path.join(CUR_DIR, os.environ['CK_DETECTIONS_OUT_DIR'])
ANNOTATIONS_OUT_DIR     = os.path.join(CUR_DIR, os.environ['CK_ANNOTATIONS_OUT_DIR'])
RESULTS_OUT_DIR         = os.path.join(CUR_DIR, os.environ['CK_RESULTS_OUT_DIR'])
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

## Processing in batches:
#
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))
SKIP_IMAGES             = int(os.getenv('CK_SKIP_IMAGES', 0))


def load_preprocessed_batch(image_list, image_index):
    batch_data = []

    for _ in range(BATCH_SIZE):
        img_file = os.path.join(IMAGE_DIR, image_list[image_index])
        img = np.fromfile(img_file, np.dtype(IMAGE_DATA_TYPE))
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3))
        if MODEL_COLOURS_BGR:
            img = img[...,::-1]     # swapping Red and Blue colour channels

        if IMAGE_DATA_TYPE != 'float32':
            img = img.astype(VECTOR_DATA_TYPE)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img/127.5 - 1.0

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img, axis=(0,1), keepdims=True)

        if MODEL_INPUT_DATA_TYPE == 'int8':
            img = np.clip(img, -128, 127)

        # Add img to batch
        batch_data.append( [img.astype(MODEL_INPUT_DATA_TYPE)] )
        image_index += 1

    nhwc_data = np.concatenate(batch_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        #print(nhwc_data.shape)
        return nhwc_data, image_index
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        #print(nchw_data.shape)
        return nchw_data, image_index


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def main():
    global BATCH_SIZE
    global BATCH_COUNT
    global MODEL_DATA_LAYOUT
    global MODEL_IMAGE_HEIGHT
    global MODEL_IMAGE_WIDTH

    setup_time_begin = time.time()

    # Load preprocessed image filenames:
    with open(IMAGE_LIST_FILE, 'r') as f:
        image_list = [s.strip() for s in f]

    # Trim the input list of preprocessed files:
    image_list = image_list[SKIP_IMAGES: BATCH_COUNT * BATCH_SIZE + SKIP_IMAGES]

    # Creating a local list of processed files and parsing it:
    image_filenames = []
    original_w_h    = []
    with open(IMAGE_LIST_FILE_NAME, 'w') as f:
        for line in image_list:
            f.write('{}\n'.format(line))
            file_name, width, height = line.split(";")
            image_filenames.append( file_name )
            original_w_h.append( (int(width), int(height)) )

    # Cleanup results directory
    if os.path.isdir(DETECTIONS_OUT_DIR):
        shutil.rmtree(DETECTIONS_OUT_DIR)
    os.mkdir(DETECTIONS_OUT_DIR)

    # Load the TensorRT model from file
    default_context = pycuda.tools.make_default_context()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    try:
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(MODEL_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            print('[TRT] successfully loaded')
    except:
        default_context.pop()
        raise RuntimeError('TensorRT model file {} is not found or corrupted'.format(MODEL_PATH))

    max_batch_size      = trt_engine.max_batch_size

    d_inputs, h_d_outputs, model_bindings = [], [], []
    for interface_layer in trt_engine:
        dtype   = trt_engine.get_binding_dtype(interface_layer)
        shape   = trt_engine.get_binding_shape(interface_layer)
        size    = trt.volume(shape) * max_batch_size

        dev_mem = cuda.mem_alloc(size * dtype.itemsize)
        model_bindings.append( int(dev_mem) )

        if trt_engine.binding_is_input(interface_layer):
            interface_type = 'Input'
            d_inputs.append(dev_mem)
            model_input_shape   = shape
        else:
            interface_type = 'Output'
            host_mem    = cuda.pagelocked_empty(size, trt.nptype(dtype))
            h_d_outputs.append({ 'host_mem': host_mem, 'dev_mem': dev_mem })
            model_output_shape  = shape
            h_output            = host_mem

        print("{} layer {}: dtype={}, shape={}, elements_per_max_batch={}".format(interface_type, interface_layer, dtype, shape, size))

    cuda_stream         = cuda.Stream()

    model_classes       = trt.volume(model_output_shape)
    labels              = load_labels(LABELS_PATH)
    num_layers          = trt_engine.num_layers
    bg_class_offset     = 1

    ## Workaround for SSD-Resnet34 model incorrectly trained on filtered labels
    class_map = None
    if (SKIPPED_CLASSES):
        class_map = []
        for i in range(len(labels) + bg_class_offset):
            if i not in SKIPPED_CLASSES:
                class_map.append(i)

    if MODEL_DATA_LAYOUT == 'NHWC':
        (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS) = model_input_shape
    else:
        (MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH) = model_input_shape

    print('Images dir: ' + IMAGE_DIR)
    print('Image list file: ' + IMAGE_LIST_FILE)
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Detections dir: ' + DETECTIONS_OUT_DIR);
    print('Normalize: {}'.format(MODEL_NORMALIZE_DATA))
    print('Subtract mean: {}'.format(SUBTRACT_MEAN))
    print('Per-channel means to subtract: {}'.format(GIVEN_CHANNEL_MEANS))

    print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
    print('Model image height: {}'.format(MODEL_IMAGE_HEIGHT))
    print('Model image width: {}'.format(MODEL_IMAGE_WIDTH))
    print('Model image channels: {}'.format(MODEL_IMAGE_CHANNELS))
    print('Model input data type: {}'.format(MODEL_INPUT_DATA_TYPE))
    print('Model (internal) data type: {}'.format(MODEL_DATA_TYPE))
    print('Model BGR colours: {}'.format(MODEL_COLOURS_BGR))
    print('Model max_batch_size: {}'.format(max_batch_size))
    print('Model num_layers: {}'.format(num_layers))
    print('Post-detection confidence score threshold: {}'.format(SCORE_THRESHOLD))
    print("")

    if BATCH_SIZE>max_batch_size:
        default_context.pop()
        raise RuntimeError("Desired batch_size ({}) exceeds max_batch_size of the model ({})".format(BATCH_SIZE,max_batch_size))

    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    total_load_time = 0
    next_batch_offset = 0
    total_inference_time = 0
    first_inference_time = 0
    images_loaded = 0

    with trt_engine.create_execution_context() as context:
        for batch_index in range(BATCH_COUNT):
            batch_number = batch_index+1
          
            begin_time = time.time()
            current_batch_offset = next_batch_offset
            batch_data, next_batch_offset = load_preprocessed_batch(image_filenames, current_batch_offset)
            vectored_batch = np.array(batch_data).ravel().astype(MODEL_INPUT_DATA_TYPE)

            load_time = time.time() - begin_time
            total_load_time += load_time
            images_loaded += BATCH_SIZE

            # Inference begins here
            begin_time = time.time()

            cuda.memcpy_htod_async(d_inputs[0], vectored_batch, cuda_stream)    # assuming one input layer for inference
            context.execute_async(bindings=model_bindings, batch_size=BATCH_SIZE, stream_handle=cuda_stream.handle)
            for output in h_d_outputs:
                cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)

            cuda_stream.synchronize()

            # Inference ends here
            inference_time = time.time() - begin_time

            print("[batch {} of {}] loading={:.2f} ms, inference={:.2f} ms".format(
                          batch_number, BATCH_COUNT, load_time*1000, inference_time*1000))

            batch_results = h_output.reshape(max_batch_size, MODEL_MAX_PREDICTIONS*7+1)[:BATCH_SIZE]

            total_inference_time += inference_time
            # Remember inference_time for the first batch
            if batch_index == 0:
                first_inference_time = inference_time

            # Process results
            for index_in_batch in range(BATCH_SIZE):
                single_image_predictions = batch_results[index_in_batch]
                num_boxes = single_image_predictions[MODEL_MAX_PREDICTIONS*7].view('int32')
                global_image_index = current_batch_offset + index_in_batch
                width_orig, height_orig = original_w_h[global_image_index]

                filename_orig = image_filenames[global_image_index]
                detections_filename = os.path.splitext(filename_orig)[0] + '.txt'
                detections_filepath = os.path.join(DETECTIONS_OUT_DIR, detections_filename)

                with open(detections_filepath, 'w') as det_file:
                    det_file.write('{:d} {:d}\n'.format(width_orig, height_orig))

                    for row in range(num_boxes):
                        (image_id, ymin, xmin, ymax, xmax, confidence, class_number) = single_image_predictions[row*7:(row+1)*7]

                        if confidence >= SCORE_THRESHOLD:
                            class_number    = int(class_number)
                            if class_map:
                                class_number = class_map[class_number]

                            image_id        = int(image_id)
                            x1              = xmin * width_orig
                            y1              = ymin * height_orig
                            x2              = xmax * width_orig
                            y2              = ymax * height_orig
                            class_label     = labels[class_number - bg_class_offset]
                            det_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {} {}\n'.format(
                                            x1, y1, x2, y2, confidence, class_number, class_label))
                
    default_context.pop()

    test_time = time.time() - test_time_begin
 
    if BATCH_COUNT > 1:
        avg_inference_time = (total_inference_time - first_inference_time) / (images_loaded - BATCH_SIZE)
    else:
        avg_inference_time = total_inference_time / images_loaded

    avg_load_time = total_load_time / images_loaded

    # Store benchmarking results:
    output_dict = {
        'run_time_state': {
            'setup_time_s': setup_time,
            'test_time_s': test_time,
            'images_load_time_total_s': total_load_time,
            'images_load_time_avg_s': avg_load_time,
            'prediction_time_total_s': total_inference_time,
            'prediction_time_avg_s': avg_inference_time,

            'avg_time_ms': avg_inference_time * 1000,
            'avg_fps': 1.0 / avg_inference_time,
            'batch_time_ms': avg_inference_time * 1000 * BATCH_SIZE,
            'batch_size': BATCH_SIZE,
        }
    }
    with open('tmp-ck-timer.json', 'w') as out_file:
        json.dump(output_dict, out_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
