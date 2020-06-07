#!/usr/bin/env python3

import json
import time
import os
import shutil
import numpy as np

from imagenet_helper import (load_preprocessed_batch, image_list, class_labels,
    MODEL_DATA_LAYOUT, MODEL_COLOURS_BGR, MODEL_INPUT_DATA_TYPE, MODEL_DATA_TYPE, MODEL_USE_DLA,
    MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_CHANNELS,
    IMAGE_DIR, IMAGE_LIST_FILE, MODEL_NORMALIZE_DATA, SUBTRACT_MEAN, GIVEN_CHANNEL_MEANS, BATCH_SIZE)

from tensorrt_helper import (initialize_predictor, inference_for_given_batch)


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))


## Writing the results out:
#
RESULTS_DIR             = os.getenv('CK_RESULTS_DIR')
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

## Processing in batches:
#
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))


def main():
    setup_time_begin = time.time()

    # Cleanup results directory
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

    pycuda_context, max_batch_size, input_volume, output_volume, num_layers = initialize_predictor()
    num_classes = len(class_labels)

    print('Images dir: ' + IMAGE_DIR)
    print('Image list file: ' + IMAGE_LIST_FILE)
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Results dir: ' + RESULTS_DIR);
    print('Normalize: {}'.format(MODEL_NORMALIZE_DATA))
    print('Subtract mean: {}'.format(SUBTRACT_MEAN))
    print('Per-channel means to subtract: {}'.format(GIVEN_CHANNEL_MEANS))

    print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
    print("DLA mode used: {}".format(MODEL_USE_DLA) )
    print('Model image height: {}'.format(MODEL_IMAGE_HEIGHT))
    print('Model image width: {}'.format(MODEL_IMAGE_WIDTH))
    print('Model image channels: {}'.format(MODEL_IMAGE_CHANNELS))
    print('Model input data type: {}'.format(MODEL_INPUT_DATA_TYPE))
    print('Model (internal) data type: {}'.format(MODEL_DATA_TYPE))
    print('Model BGR colours: {}'.format(MODEL_COLOURS_BGR))
    print('Model max_batch_size: {}'.format(max_batch_size))
    print('Model output volume (number of outputs per one prediction): {}'.format(output_volume))
    print('Model num_layers: {}'.format(num_layers))
    print('Number of class_labels: {}'.format(num_classes))
    print("")


    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    image_index = 0
    total_load_time = 0
    total_classification_time = 0
    first_classification_time = 0
    images_loaded = 0

    for batch_index in range(BATCH_COUNT):
        batch_number = batch_index+1

        begin_time = time.time()
        batch_data, image_index = load_preprocessed_batch(image_list, image_index)

        load_time = time.time() - begin_time
        total_load_time += load_time
        images_loaded += BATCH_SIZE

        trimmed_batch_results, inference_time_s = inference_for_given_batch(batch_data)

        print("[batch {} of {}] loading={:.2f} ms, inference={:.2f} ms".format(
                      batch_number, BATCH_COUNT, load_time*1000, inference_time_s*1000))

        total_classification_time += inference_time_s
        # Remember first batch prediction time
        if batch_index == 0:
            first_classification_time = inference_time_s

        # Process results
        for index_in_batch in range(BATCH_SIZE):
            one_batch_result = trimmed_batch_results[index_in_batch]
            if output_volume==1:
                arg_max = one_batch_result[0]
                softmax_vector = [0]*arg_max + [1] + [0]*(num_classes-arg_max-1)
            else:
                softmax_vector = one_batch_result[-num_classes:]    # skipping the background class on the left (if present)
            global_index = batch_index * BATCH_SIZE + index_in_batch
            res_file = os.path.join(RESULTS_DIR, image_list[global_index])
            with open(res_file + '.txt', 'w') as f:
                for prob in softmax_vector:
                    f.write('{}\n'.format(prob))
                
    pycuda_context.pop()

    test_time = time.time() - test_time_begin
 
    if BATCH_COUNT > 1:
        avg_classification_time = (total_classification_time - first_classification_time) / (images_loaded - BATCH_SIZE)
    else:
        avg_classification_time = total_classification_time / images_loaded

    avg_load_time = total_load_time / images_loaded

    # Store benchmarking results:
    output_dict = {
        'setup_time_s': setup_time,
        'test_time_s': test_time,
        'images_load_time_total_s': total_load_time,
        'images_load_time_avg_s': avg_load_time,
        'prediction_time_total_s': total_classification_time,
        'prediction_time_avg_s': avg_classification_time,

        'avg_time_ms': avg_classification_time * 1000,
        'avg_fps': 1.0 / avg_classification_time,
        'batch_time_ms': avg_classification_time * 1000 * BATCH_SIZE,
        'batch_size': BATCH_SIZE,
    }
    with open('tmp-ck-timer.json', 'w') as out_file:
        json.dump(output_dict, out_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
