Usage examples:

```bash
    ck detect soft:model.tensorrt --full_path=/full/path/to/ResNet50_model_fp32.trt \
        --extra_tags=fp32,resnet,resnet50,image-classification,converted-from-caffe \
        --ienv.ML_MODEL_DATA_TYPE=fp32 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_COLOUR_CHANNELS_BGR=YES \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="123.68 116.6 103.94"
```

```bash
    ck detect soft:model.tensorrt --full_path=/full/path/to/resnet-MultiStream-dla-b15-int8.plan \
        --extra_tags=maxbatch.15,int8,resnet,resnet50,dla,image-classification,converted-by-nvidia \
        --ienv.ML_MODEL_DATA_TYPE=int8 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="128 128 128" \
        --ienv.ML_MODEL_MAX_BATCH_SIZE=15
```

