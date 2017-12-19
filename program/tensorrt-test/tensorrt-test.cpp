/*
 * Loosely based on http://github.com/dusty-nv/jetson-inference
 *
 * Therefore, assuming:
 * 2016 (c) NVIDIA
 * 2017 (c) dividiti
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"

#include <string>

#include <dirent.h>

#define DEFAULT_BATCH_SIZE 1

int classifyImageRGBA(imageNet* net, const char* imgPath)
{
    int exit_status = EXIT_SUCCESS;

    // Load image from disk.
    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    int    imgWidth  = 0;
    int    imgHeight = 0;

    if( !loadImageRGBA(imgPath, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
    {
        printf("[tensorrt-test] Failed to load image '%s'\n", imgPath);
        exit_status = EXIT_FAILURE;
    }
    else
    {
        // Classify image.
        float confidence = 0.0f;
        const int imgClass = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

        if( imgClass < 0 )
        {
            printf("[tensorrt-test] Failed to classify '%s'  (result=%i)\n", imgPath, imgClass);
            exit_status = EXIT_FAILURE;
        }
        else
        {
            printf("[tensorrt-test]  '%s' -> %2.5f%% class #%i (%s)\n", imgPath, confidence * 100.0f, imgClass, net->GetClassDesc(imgClass));
        }
        CUDA(cudaFreeHost(imgCPU));
    }

    return exit_status;
}


// Main entry point.
int main( int argc, char** argv )
{
    int exit_status = EXIT_SUCCESS;

    // Print environment variables set by CK.
    printf("\n[tensorrt-test] CK settings detected:\n");

    const char * caffe_model_var = "CK_CAFFE_MODEL";
    const char * caffe_model_val = getenv(caffe_model_var);
    printf("     %s=\"%s\"\n", caffe_model_var,
                               caffe_model_val ? caffe_model_val : "?");

    const char * caffe_weights_var = "CK_ENV_MODEL_CAFFE_WEIGHTS";
    const char * caffe_weights_val = getenv(caffe_weights_var);
    printf("     %s=\"%s\"\n", caffe_weights_var,
                               caffe_weights_val ? caffe_weights_val : "?");

    const char * imagenet_val_dir_var = "CK_ENV_DATASET_IMAGENET_VAL";
    const char * imagenet_val_dir_val = getenv(imagenet_val_dir_var);
    printf("     %s=\"%s\"\n", imagenet_val_dir_var,
                               imagenet_val_dir_val ? imagenet_val_dir_val : "?");

    const char * imagenet_mean_bin_var = "CK_CAFFE_IMAGENET_MEAN_BIN";
    const char * imagenet_mean_bin_val = getenv(imagenet_mean_bin_var);
    printf("     %s=\"%s\"\n", imagenet_mean_bin_var,
                               imagenet_mean_bin_val ? imagenet_mean_bin_val : "?");

    const char * imagenet_synset_words_txt_var = "CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT";
    const char * imagenet_synset_words_txt_val = getenv(imagenet_synset_words_txt_var);
    printf("     %s=\"%s\"\n", imagenet_synset_words_txt_var,
                               imagenet_synset_words_txt_val ? imagenet_synset_words_txt_val : "?");

    const char * imagenet_val_txt_var = "CK_CAFFE_IMAGENET_VAL_TXT";
    const char * imagenet_val_txt_val = getenv(imagenet_val_txt_var);
    printf("     %s=\"%s\"\n", imagenet_val_txt_var,
                               imagenet_val_txt_val ? imagenet_val_txt_val : "?");

    const char * tensorrt_max_num_images_var = "CK_TENSORRT_MAX_NUM_IMAGES";
    const char * tensorrt_max_num_images_val = getenv(tensorrt_max_num_images_var);
    printf("     %s=\"%s\"\n", tensorrt_max_num_images_var,
                            tensorrt_max_num_images_val ? tensorrt_max_num_images_val : "?");

    const char * tensorrt_enable_fp16_var = "CK_TENSORRT_ENABLE_FP16";
    const char * tensorrt_enable_fp16_val = getenv(tensorrt_enable_fp16_var);
    printf("     %s=\"%s\"\n", tensorrt_enable_fp16_var,
                               tensorrt_enable_fp16_val ? tensorrt_enable_fp16_val : "?");

    // Print configuration variables inferred.
    printf("\n[tensorrt-test] TensorRT settings inferred:\n");
    const size_t tensorrt_max_num_images = tensorrt_max_num_images_val ? atoi(tensorrt_max_num_images_val) : 50000;
    printf("     TENSORRT_MAX_NUM_IMAGES=%ld\n", tensorrt_max_num_images);

    const bool   tensorrt_enable_fp16 = tensorrt_enable_fp16_val ? (bool)atoi(tensorrt_enable_fp16_val) : true;
    printf("     TENSORRT_ENABLE_FP16=%d\n", tensorrt_enable_fp16);

    // for classification default batch size is 1
    const uint32_t maxBatchSize = DEFAULT_BATCH_SIZE;

    // Print command line arguments.
    printf("\n[tensorrt-test] Command line arguments (%i):", argc);
    for( int i = 0; i < argc; ++i )
        printf("\n     [%i] %s", i, argv[i]);
    printf("\n");

    // Clean possibly cached TensorRT model.
    printf("\n[tensorrt-test] Cleaning TensorRT model cache...");
    {
        const char* cache_ext = "tensorcache";
        char* cache_path = (char*) malloc(strlen(caffe_weights_val) + strlen(cache_ext) + 2);
        sprintf(cache_path, "%s.%s", caffe_weights_val, cache_ext);
        printf("\n[tensorrt-test] - file \'%s\' removed ", cache_path);
        int status = remove(cache_path);
        if (0 == status)
        {
            printf("successfully!\n");
        }
        else
        {
            printf("unsuccessfully!\n");
        }
        free(cache_path);
    }

    printf("\n[tensorrt-test] Start imageNet::Create...");
    // Create an imageNet object.
    imageNet* net = imageNet::Create(
                        caffe_model_val,
                        caffe_weights_val,
                        imagenet_mean_bin_val,
                        imagenet_synset_words_txt_val,
                        "data", "prob",
                        maxBatchSize
                    );

#if( 1 == CK_TENSORRT_ENABLE_PROFILER )
    net->EnableProfiler();
#endif

    if( !net )
    {
        printf("\n[tensorrt-test] Failed to create ImageNet classifier\n");
        return EXIT_FAILURE;
    }

    // Classify a single image or all images in $CK_ENV_DATASET_IMAGENET_VAL.
    if( argc == 2 )
    {
        const char* imgPath = argv[1];
        exit_status = classifyImageRGBA(net, imgPath);
    }
    else if( argc == 1 )
    {
        DIR* dir;
        struct dirent* ent;
        if( (dir = opendir(imagenet_val_dir_val)) )
        {
            const char* sample_imagenet_val_file = "ILSVRC2012_val_00002212.JPEG"; // 00002212 with AlexNet: top1="no", top5="yes"
            char* imagenet_val_path = (char*) malloc(strlen(imagenet_val_dir_val) + strlen(sample_imagenet_val_file) + 2);
            size_t num_images = 0;

            printf("\n[tensorrt-test] Scanning directory: %s\n", imagenet_val_path);
            while( (ent = readdir(dir)) && (num_images < tensorrt_max_num_images) )
            {
                const char* imagenet_val_file = ent->d_name;
                if( strlen(imagenet_val_file) < strlen(sample_imagenet_val_file) )
                {
                    // Skip '.' and '..'.
                    continue;
                }
                printf("\n[tensorrt-test] Classifying image #%ld out of %ld\n", num_images+1, tensorrt_max_num_images);
                sprintf(imagenet_val_path, "%s/%s", imagenet_val_dir_val, imagenet_val_file);
                exit_status = classifyImageRGBA(net, imagenet_val_path);
                if (exit_status == EXIT_FAILURE)
                {
                    return exit_status;
                }
                num_images++;
            }
            closedir(dir);
            free(imagenet_val_path);
        }
        else
        {
            printf("\n[tensorrt-test] Failed to open directory \'%s\'\n", imagenet_val_dir_var);
            exit_status = EXIT_FAILURE;
        }
    }
    else
    {
        printf("\n[tensorrt-test] Usage: %s [path]", argv[0]);
        printf(" (by default, all files in \'%s\' dir)\n", imagenet_val_dir_val);
        exit_status = EXIT_FAILURE;
    }

    printf("\n[tensorrt-test] Shutting down...\n");
    delete net;

    return exit_status;
}
