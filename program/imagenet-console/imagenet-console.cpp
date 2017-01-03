/*
 * Loosely based on http://github.com/dusty-nv/jetson-inference
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"

#include <string>

#include <dirent.h>

int classifyImageRGBA(imageNet* net, const char* imgPath)
{
    int exit_status = EXIT_SUCCESS;

    // load image from disk
    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    int    imgWidth  = 0;
    int    imgHeight = 0;

    if( !loadImageRGBA(imgPath, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
    {
        printf("[imagenet-console]  failed to load image '%s'\n", imgPath);
        exit_status = EXIT_FAILURE;
    }
    else
    {
        // classify image
        float confidence = 0.0f;
        const int imgClass = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

        if( imgClass < 0 )
        {
            printf("[imagenet-console]  failed to classify '%s'  (result=%i)\n", imgPath, imgClass);
            exit_status = EXIT_FAILURE;
        }
        else
        {
            printf("[imagenet-console]  '%s' -> %2.5f%% class #%i (%s)\n", imgPath, confidence * 100.0f, imgClass, net->GetClassDesc(imgClass));
        }
        CUDA(cudaFreeHost(imgCPU));
    }

    return exit_status;
}


// main entry point
int main( int argc, char** argv )
{
    int exit_status = EXIT_SUCCESS;

    // print environment variables set by CK
    printf("\n[imagenet-console]  ck-env:\n");

    const char * caffe_model_var = "CK_CAFFE_MODEL";
    const char * caffe_model_val = getenv(caffe_model_var);
    printf("     %s=\"%s\"\n", caffe_model_var, caffe_model_val ? caffe_model_val : "?");

    const char * caffe_weights_var = "CK_ENV_MODEL_CAFFE_WEIGHTS";
    const char * caffe_weights_val = getenv(caffe_weights_var);
    printf("     %s=\"%s\"\n", caffe_weights_var, caffe_weights_val ? caffe_weights_val : "?");

    const char * imagenet_val_dir_var = "CK_ENV_DATASET_IMAGENET_VAL";
    const char * imagenet_val_dir_val = getenv(imagenet_val_dir_var);
    printf("     %s=\"%s\"\n", imagenet_val_dir_var, imagenet_val_dir_val ? imagenet_val_dir_val : "?");

    const char * imagenet_mean_bin_var = "CK_CAFFE_IMAGENET_MEAN_BIN";
    const char * imagenet_mean_bin_val = getenv(imagenet_mean_bin_var);
    printf("     %s=\"%s\"\n", imagenet_mean_bin_var, imagenet_mean_bin_val ? imagenet_mean_bin_val : "?");

    const char * imagenet_synset_words_txt_var = "CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT";
    const char * imagenet_synset_words_txt_val = getenv(imagenet_synset_words_txt_var);
    printf("     %s=\"%s\"\n", imagenet_synset_words_txt_var, imagenet_synset_words_txt_val ? imagenet_synset_words_txt_val : "?");

    const char * imagenet_val_txt_var = "CK_CAFFE_IMAGENET_VAL_TXT";
    const char * imagenet_val_txt_val = getenv(imagenet_val_txt_var);
    printf("     %s=\"%s\"\n", imagenet_val_txt_var, imagenet_val_txt_val ? imagenet_val_txt_val : "?");

    const char * caffe_iterations_var = "CK_CAFFE_ITERATIONS";
    const char * caffe_iterations_val = getenv(caffe_iterations_var);
    printf("     %s=%s\n", caffe_iterations_var, caffe_iterations_val ? caffe_iterations_val : "?");

    const char * caffe_batch_size_var = "CK_CAFFE_BATCH_SIZE";
    const char * caffe_batch_size_val = getenv(caffe_batch_size_var);
    printf("     %s=%s\n", caffe_batch_size_var, caffe_batch_size_val ? caffe_batch_size_val : "?");

    const char * tensorrt_max_images_var = "CK_TENSORRT_MAX_IMAGES";
    const char * tensorrt_max_images_val = getenv(tensorrt_max_images_var);
    const size_t max_images = tensorrt_max_images_val ? std::stoi(tensorrt_max_images_val) : 1;
    printf("     %s=%ld\n", tensorrt_max_images_var, max_images);


    // print command line arguments
    printf("\n[imagenet-console]  args (%i):", argc);

    for( int i = 0; i < argc; i++ )
        printf("\n     [%i] %s", i, argv[i]);

    printf("\n\n");

    // create imageNet
    imageNet* net = imageNet::Create(
                        caffe_model_val,
                        caffe_weights_val,
                        imagenet_mean_bin_val,
                        imagenet_synset_words_txt_val
                    );
#if( 1 == CK_TENSORRT_ENABLE_PROFILER )
    net->EnableProfiler();
#endif

    if( !net )
    {
        printf("\n[imagenet-console]  failed to initialize imageNet\n");
        return EXIT_FAILURE;
    }

    // classify a single image or all images in $CK_ENV_DATASET_IMAGENET_VAL
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
            while( (ent = readdir(dir)) && (num_images < max_images) )
            {
                const char* imagenet_val_file = ent->d_name;
                if( strlen(imagenet_val_file) < strlen(sample_imagenet_val_file) )
                {
                    // skip '.' and '..'
                    continue;
                }
                printf("\n[imagenet-console]  classifying image #%ld out of %ld\n", num_images+1, max_images);
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
        } else {
            printf("\n[imagenet-console]  failed to open directory \'%s\'\n", imagenet_val_dir_var);
            exit_status = EXIT_FAILURE;
        }
    }
    else
    {
        printf("\n[imagenet-console]  usage: %s [path]", argv[0]);
        printf(" (by default, all files in \'%s\' dir)\n", imagenet_val_dir_val);
        exit_status = EXIT_FAILURE;
    }

    printf("\n[imagenet-console]  shutting down...\n");
    delete net;
    return exit_status;
}
