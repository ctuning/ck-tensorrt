Specific usage examples:

```bash
    ck detect soft:model.tensorrt --full_path=/datasets/tensorrt_plans_for_Xavier/ssd-small/MultiStream/ssd-small-MultiStream-gpu-b20-fp32.plan \
        --extra_tags=maxbatch.20,fp32,ssd-mobilenet,gpu,object-detection,converted-by-nvidia \
        --cus.version=ssd-mobilenet_nvidia_fp32 \
        --ienv.ML_MODEL_CLASS_LABELS=/datasets/tensorrt_plans_for_Xavier/flatlabels.txt \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_IMAGE_HEIGHT=300 \
        --ienv.ML_MODEL_IMAGE_WIDTH=300 \
        --ienv.ML_MODEL_INPUT_DATA_TYPE=float32 \
        --ienv.ML_MODEL_DATA_TYPE=float32 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=YES \
        --ienv.ML_MODEL_SUBTRACT_MEAN=NO \
        --ienv.ML_MODEL_MAX_PREDICTIONS=100 \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=20
```

```bash
    ck detect soft:model.tensorrt --full_path=/datasets/tensorrt_plans_for_Xavier/ssd-small/MultiStream/ssd-small-MultiStream-gpu-b20-int8_linear.plan \
        --extra_tags=maxbatch.20,int8,linear,ssd-mobilenet,gpu,object-detection,converted-by-nvidia \
        --cus.version=ssd-mobilenet_nvidia_int8_linear \
        --ienv.ML_MODEL_CLASS_LABELS=/datasets/tensorrt_plans_for_Xavier/flatlabels.txt \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_IMAGE_HEIGHT=300 \
        --ienv.ML_MODEL_IMAGE_WIDTH=300 \
        --ienv.ML_MODEL_INPUT_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_MAX_PREDICTIONS=100 \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="128 128 128" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=20
```


```bash
    ck detect soft:model.tensorrt --full_path=/datasets/tensorrt_plans_for_Xavier/ssd-large/MultiStream/ssd-large-MultiStream-gpu-b2-int8.plan \
        --extra_tags=maxbatch.2,int8,ssd-resnet,object-detection,converted-by-nvidia \
        --cus.version=ssd-resnet_nvidia_int8 \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_IMAGE_HEIGHT=1200 \
        --ienv.ML_MODEL_IMAGE_WIDTH=1200 \
        --ienv.ML_MODEL_INPUT_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="123.68 116.78 103.94" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=2 \
        --ienv.ML_MODEL_MAX_PREDICTIONS=200 \
        --ienv.ML_MODEL_CLASS_LABELS=/datasets/tensorrt_plans_for_Xavier/flatlabels.txt \
        --ienv.ML_MODEL_SKIPS_ORIGINAL_DATASET_CLASSES=12,26,29,30,45,66,68,69,71,83 \
        --ienv.ML_MODEL_TENSORRT_PLUGIN=/datasets/tensorrt_plans_for_Xavier/libnmsoptplugin.so
```

```bash
    ck detect soft:model.tensorrt --full_path=/datasets/tensorrt_plans_for_Xavier/ssd-large-MultiStream-gpu-b2-fp16.plan \
        --extra_tags=maxbatch.2,fp16,ssd-resnet,object-detection,converted-by-nvidia \
        --cus.version=ssd-resnet_nvidia_fp16 \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_IMAGE_HEIGHT=1200 \
        --ienv.ML_MODEL_IMAGE_WIDTH=1200 \
        --ienv.ML_MODEL_INPUT_DATA_TYPE=float32 \
        --ienv.ML_MODEL_DATA_TYPE=float16 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=YES \
        --ienv.ML_MODEL_NORMALIZE_LOWER=0.0 \
        --ienv.ML_MODEL_NORMALIZE_UPPER=1.0 \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="0.485 0.456 0.406" \
        --ienv.ML_MODEL_GIVEN_CHANNEL_STDS="0.229 0.224 0.225" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=2 \
        --ienv.ML_MODEL_MAX_PREDICTIONS=200 \
        --ienv.ML_MODEL_CLASS_LABELS=/datasets/tensorrt_plans_for_Xavier/flatlabels.txt \
        --ienv.ML_MODEL_SKIPS_ORIGINAL_DATASET_CLASSES=12,26,29,30,45,66,68,69,71,83 \
        --ienv.ML_MODEL_TENSORRT_PLUGIN=/datasets/tensorrt_plans_for_Xavier/libnmsoptplugin.so
```

```bash
    ck detect soft:model.tensorrt --full_path=/datasets/tensorrt_plans_for_Xavier/ssd-large-MultiStream-gpu-b2-fp32.plan \
        --extra_tags=maxbatch.2,fp32,ssd-resnet,object-detection,converted-by-nvidia \
        --cus.version=ssd-resnet_nvidia_fp32 \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_IMAGE_HEIGHT=1200 \
        --ienv.ML_MODEL_IMAGE_WIDTH=1200 \
        --ienv.ML_MODEL_INPUT_DATA_TYPE=float32 \
        --ienv.ML_MODEL_DATA_TYPE=float32 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=YES \
        --ienv.ML_MODEL_NORMALIZE_LOWER=0.0 \
        --ienv.ML_MODEL_NORMALIZE_UPPER=1.0 \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="0.485 0.456 0.406" \
        --ienv.ML_MODEL_GIVEN_CHANNEL_STDS="0.229 0.224 0.225" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=2 \
        --ienv.ML_MODEL_MAX_PREDICTIONS=200 \
        --ienv.ML_MODEL_CLASS_LABELS=/datasets/tensorrt_plans_for_Xavier/flatlabels.txt \
        --ienv.ML_MODEL_SKIPS_ORIGINAL_DATASET_CLASSES=12,26,29,30,45,66,68,69,71,83 \
        --ienv.ML_MODEL_TENSORRT_PLUGIN=/datasets/tensorrt_plans_for_Xavier/libnmsoptplugin.so
```

