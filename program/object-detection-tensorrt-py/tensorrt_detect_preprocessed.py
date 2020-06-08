#!/usr/bin/env python3

import json
import time
import os
import shutil
import numpy as np

from coco_helper import (load_preprocessed_batch, image_filenames, original_w_h,
    class_labels, num_classes, bg_class_offset, class_map,
    MODEL_DATA_LAYOUT, MODEL_COLOURS_BGR, MODEL_INPUT_DATA_TYPE, MODEL_DATA_TYPE, MODEL_USE_DLA,
    MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_CHANNELS,
    IMAGE_DIR, IMAGE_LIST_FILE, MODEL_NORMALIZE_DATA, SUBTRACT_MEAN, GIVEN_CHANNEL_MEANS, BATCH_SIZE, BATCH_COUNT)

from tensorrt_helper import (initialize_predictor, inference_for_given_batch)


## Post-detection filtering by confidence score:
#
SCORE_THRESHOLD = float(os.getenv('CK_DETECTION_THRESHOLD', 0.0))


## Model properties:
#
MODEL_MAX_PREDICTIONS   = int(os.getenv('ML_MODEL_MAX_PREDICTIONS', 100))


## Writing the results out:
#
CUR_DIR = os.getcwd()
DETECTIONS_OUT_DIR      = os.path.join(CUR_DIR, os.environ['CK_DETECTIONS_OUT_DIR'])
ANNOTATIONS_OUT_DIR     = os.path.join(CUR_DIR, os.environ['CK_ANNOTATIONS_OUT_DIR'])
RESULTS_OUT_DIR         = os.path.join(CUR_DIR, os.environ['CK_RESULTS_OUT_DIR'])
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')


def main():
    setup_time_begin = time.time()

    # Cleanup results directory
    if os.path.isdir(DETECTIONS_OUT_DIR):
        shutil.rmtree(DETECTIONS_OUT_DIR)
    os.mkdir(DETECTIONS_OUT_DIR)

    pycuda_context, max_batch_size, input_volume, output_volume, num_layers = initialize_predictor()

    print('Images dir: ' + IMAGE_DIR)
    print('Image list file: ' + IMAGE_LIST_FILE)
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Detections dir: ' + DETECTIONS_OUT_DIR);
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
    print('Post-detection confidence score threshold: {}'.format(SCORE_THRESHOLD))
    print("")

    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    total_load_time = 0
    next_batch_offset = 0
    total_inference_time = 0
    first_inference_time = 0
    images_loaded = 0

    for batch_index in range(BATCH_COUNT):
        batch_number = batch_index+1

        begin_time = time.time()
        current_batch_offset = next_batch_offset
        batch_data, next_batch_offset = load_preprocessed_batch(image_filenames, current_batch_offset)

        load_time = time.time() - begin_time
        total_load_time += load_time
        images_loaded += BATCH_SIZE

        trimmed_batch_results, inference_time_s = inference_for_given_batch(batch_data)

        print("[batch {} of {}] loading={:.2f} ms, inference={:.2f} ms".format(
                      batch_number, BATCH_COUNT, load_time*1000, inference_time_s*1000))

        total_inference_time += inference_time_s
        # Remember inference_time for the first batch
        if batch_index == 0:
            first_inference_time = inference_time_s

        # Process results
        for index_in_batch in range(BATCH_SIZE):
            single_image_predictions = trimmed_batch_results[index_in_batch]
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
                        class_label     = class_labels[class_number - bg_class_offset]
                        det_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {} {}\n'.format(
                                        x1, y1, x2, y2, confidence, class_number, class_label))
                
    pycuda_context.pop()

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
