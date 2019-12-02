#!/bin/bash

read -d '' CMD <<END_OF_CMD
    $CK_ENV_COMPILER_PYTHON_FILE $PACKAGE_DIR/tf2tensorrt_model_converter.py "${CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH}" "${INSTALL_DIR}/${PACKAGE_NAME}" --model_data_layout="${ML_MODEL_DATA_LAYOUT}" --input_layer_name="${CK_ENV_TENSORFLOW_MODEL_INPUT_LAYER_NAME}" --input_height="${CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT}" --input_width="${CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH}" --output_layer_name="${CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME}" --output_data_type="${ML_MODEL_DATA_TYPE}" --max_batch_size="${ML_MODEL_MAX_BATCH_SIZE}"
END_OF_CMD

echo ${CMD}

eval ${CMD}
