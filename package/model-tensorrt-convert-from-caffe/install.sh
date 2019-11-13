#!/bin/bash

$CK_ENV_COMPILER_PYTHON_FILE $PACKAGE_DIR/caffe2tensorrt_model_converter.py "${CK_ENV_MODEL_CAFFE_WEIGHTS}" "${CK_ENV_MODEL_CAFFE_DEPLOY}" "${INSTALL_DIR}/${PACKAGE_NAME}" --output_data_type "${ML_MODEL_DATA_TYPE}" --max_batch_size "${ML_MODEL_MAX_BATCH_SIZE}"
