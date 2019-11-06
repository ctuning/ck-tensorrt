Usage example:

```bash
    ck detect soft:model.tensorrt --full_path=/full/path/to/ResNet50_model_fp32.trt \
        --extra_tags=fp32,resnet,resnet50,image-classification \
        --ienv.ML_MODEL_DATA_TYPE=fp32 \
        --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
        --ienv.ML_MODEL_NORMALIZE_DATA=NO \
        --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
        --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS="123.68 116.6 103.94"
```

