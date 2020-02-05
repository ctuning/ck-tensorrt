Usage examples:

```bash
    ck detect soft:model.tensorrt --full_path=/full/path/to/ResNet50_model_fp32.trt \
        --extra_tags=fp32,resnet,resnet50,image-classification,converted-from-caffe \
        --cus.version=resnet_caffe_fp32 \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=YES \
        --ienv.ML_MODEL_MODEL_IMAGE_HEIGHT=224 \
        --ienv.ML_MODEL_MODEL_IMAGE_WIDTH=224 \
        --ienv.ML_MODEL_DATA_TYPE=float32 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="123.68 116.78 103.94"
```

```bash
    ck detect soft:model.tensorrt --full_path=/full/path/to/resnet-MultiStream-dla-b15-int8.plan \
        --extra_tags=maxbatch.15,int8,resnet,resnet50,dla,image-classification,converted-by-nvidia \
        --cus.version=resnet_nvidia_int8 \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_MODEL_IMAGE_HEIGHT=224 \
        --ienv.ML_MODEL_MODEL_IMAGE_WIDTH=224 \
        --ienv.ML_MODEL_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="123.68 116.78 103.94" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=15
```

```bash
    ck detect soft:model.tensorrt --full_path=/full/path/to/mobilenet-MultiStream-gpu-b250-int8.plan \
        --extra_tags=maxbatch.250,int8,mobilenet,gpu,image-classification,converted-by-nvidia \
        --cus.version=mobilenet_nvidia_int8 \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=NO \
        --ienv.ML_MODEL_MODEL_IMAGE_HEIGHT=224 \
        --ienv.ML_MODEL_MODEL_IMAGE_WIDTH=224 \
        --ienv.ML_MODEL_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="128 128 128" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=250
```

