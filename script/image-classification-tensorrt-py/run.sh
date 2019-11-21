#!/bin/bash

task="image-classification"
imagenet_size=50000

# Implementations.
implementation_tensorrt="image-classification-tensorrt-py"
implementations=( "${implementation_tensorrt}" )

# Modes.
modes=( "performance" "accuracy" )

# System.
hostname=`hostname`
if [ "${hostname}" = "tx1" ]; then
  device="${hostname}"
  library="py-tensorrt-5.1.6.1"
  library_tags="lib,python-package,tensorrt,5.1.6.1"
elif [ "${hostname}" = "velociti" ]; then
  device="gtx1080"
  library="py-tensorrt-5.1.5.0"
  library_tags="lib,python-package,tensorrt,5.1.5.0"
else
  device="${hostname}"
  library="py-tensorrt"
  library_tags="lib,python-package,tensorrt"
fi

# Compiler.
if [ "${device}" = "tx1" ]; then
  compiler_tags="gcc,v7"
else
  compiler_tags="gcc"
fi

# Models.
models=( "resnet" )
models_tags=( "model,tensorrt,converted-from-onnx,resnet" )
models_preprocessing_tags=( "dataset,side.224,crop.875,inter.linear,preprocessed,using-opencv" )

# Numerical data types.
data_types=( "fp16" "fp32" )

# Max batch sizes.
max_batch_sizes=$(seq 1 20)


experiment_id=1
# Iterate for each implementation.
for implementation in ${implementations[@]}; do
  if [ "${implementation}" != "${implementation_tensorrt}" ]; then
    echo "ERROR: Unsupported implementation '${implementation}'!"
    exit 1
  fi

  # Iterate for each model.
  for i in $(seq 1 ${#models[@]}); do
    # Configure the model.
    model=${models[${i}-1]}
    model_tags=${models_tags[${i}-1]}
    model_preprocessing_tags=${models_preprocessing_tags[${i}-1]}

    # Iterate for each data type.
    for data_type in ${data_types[@]}; do

      # Iterate for each mode.
      for mode in ${modes[@]}; do
        # TODO: Use the maximum batch size for accuracy experiments.
        if [ "${mode}" == "accuracy" ]; then continue; fi

        # Iterate for each max batch size.
        for max_batch_size in ${max_batch_sizes[@]}; do

          # Iterate for each batch size up to max batch size.
          for batch_size in $(seq 1 ${max_batch_size}); do

            # Configure record settings.
            record_uoa="${task}.${device}.${library}.${model}.${data_type}.max-batch-${max_batch_size}.batch-${batch_size}.${mode}"
            record_tags="${task},${device},${library},${model},${data_type},max-batch-${max_batch_size},batch-${batch_size},${mode}"
            if [ "${mode}" = "accuracy" ]; then
              # Get substring after "preprocessed," to end.
              preprocessing="${model_preprocessing_tags##*preprocessed,}"
              record_uoa+=".${preprocessing}"
              record_tags+=",${preprocessing}"
            fi
  
            echo "[`date`] Experiment #"${experiment_id}": ${record_uoa} ..."
            experiment_id=$((${experiment_id}+1))
  
            # Skip automatically if experiment record already exists.
            record_dir=$(ck list local:experiment:${record_uoa})
            if [ "${record_dir}" != "" ]; then
              echo "[`date`] - skipping ..."
              echo
              continue
            fi
  
            # Skip manually.
            if [ "${implementation}" != "${implementation_tensorrt}" ] ; then
              echo "[`date`] - skipping ..."
              echo
              continue
            fi
  
            # Run (but before that print the exact command we are about to run).
            echo "[`date`] - running ..."
            read -d '' CMD <<END_OF_CMD
            ck benchmark program:${implementation} \
            --speed --repetitions=10 \
            --env.CK_BATCH_SIZE=${batch_size} \
            --dep_add_tags.weights=${model_tags},${data_type},maxbatch.${max_batch_size} \
            --dep_add_tags.images=${model_preprocessing_tags} \
            --dep_add_tags.compiler=${compiler_tags} \
            --dep_add_tags.python=v3 \
            --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
            --skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
            echo ${CMD}
            eval ${CMD}
            # Check for errors.
            if [ "${?}" != "0" ]; then
              echo "ERROR: Failed running '${model}' with '${implementation}'!"
              exit 1
            fi
            echo
          done # for each batch size up to max batch size
        done # for each max batch size
      done # for each mode
    done # for each data type
  done # for each model
done # for each implementation
echo "[`date`] Done."
