/*
 * Based on http://github.com/dusty-nv/jetson-inference
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"

#include <string>


int classifyImageRGBA(imageNet* net, const char* imgPath)
{
	// load image from disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;

	if( !loadImageRGBA(imgPath, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgPath);
		return 1;
	}

	float confidence = 0.0f;

	// classify image
	const int imgClass = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

	if( imgClass < 0 )
	{
		printf("imagenet-console:  failed to classify '%s'  (result=%i)\n", imgPath, imgClass);
	}
	else
	{
		printf("imagenet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgPath, confidence * 100.0f, imgClass, net->GetClassDesc(imgClass));
	}

	CUDA(cudaFreeHost(imgCPU));

	return 0;
}


// main entry point
int main( int argc, char** argv )
{
	// print environment variables set by CK
	printf("[imagenet-console]  ck-env:\n");

	const char * caffe_model_var = "CK_CAFFE_MODEL";
	const char * caffe_model_val = getenv(caffe_model_var);
	printf("    %s=\"%s\"\n", caffe_model_var, caffe_model_val ? caffe_model_val : "?");

	const char * caffe_weights_var = "CK_ENV_MODEL_CAFFE_WEIGHTS";
	const char * caffe_weights_val = getenv(caffe_weights_var);
	printf("    %s=\"%s\"\n", caffe_weights_var, caffe_weights_val ? caffe_weights_val : "?");

	const char * imagenet_val_dir_var = "CK_ENV_DATASET_IMAGENET_VAL";
	const char * imagenet_val_dir_val = getenv(imagenet_val_dir_var);
	printf("    %s=\"%s\"\n", imagenet_val_dir_var, imagenet_val_dir_val ? imagenet_val_dir_val : "?");

	const char * imagenet_mean_bin_var = "CK_CAFFE_IMAGENET_MEAN_BIN";
	const char * imagenet_mean_bin_val = getenv(imagenet_mean_bin_var);
	printf("    %s=\"%s\"\n", imagenet_mean_bin_var, imagenet_mean_bin_val ? imagenet_mean_bin_val : "?");

	const char * imagenet_synset_words_txt_var = "CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT";
	const char * imagenet_synset_words_txt_val = getenv(imagenet_synset_words_txt_var);
	printf("    %s=\"%s\"\n", imagenet_synset_words_txt_var, imagenet_synset_words_txt_val ? imagenet_synset_words_txt_val : "?");

	const char * imagenet_val_txt_var = "CK_CAFFE_IMAGENET_VAL_TXT";
	const char * imagenet_val_txt_val = getenv(imagenet_val_txt_var);
	printf("    %s=\"%s\"\n", imagenet_val_txt_var, imagenet_val_txt_val ? imagenet_val_txt_val : "?");

	const char * caffe_iterations_var = "CK_CAFFE_ITERATIONS";
	const char * caffe_iterations_val = getenv(caffe_iterations_var);
	printf("    %s=%s\n", caffe_iterations_var, caffe_iterations_val ? caffe_iterations_val : "?");

	const char * caffe_batch_size_var = "CK_CAFFE_BATCH_SIZE";
	const char * caffe_batch_size_val = getenv(caffe_batch_size_var);
	printf("    %s=%s\n", caffe_batch_size_var, caffe_batch_size_val ? caffe_batch_size_val : "?");

	const char * tensorrt_max_images_var = "CK_TENSORRT_MAX_IMAGES";
	const char * tensorrt_max_images_val = getenv(tensorrt_max_images_var);
	const size_t max_images = tensorrt_max_images_val ? std::stoi(tensorrt_max_images_val) : 1;
	printf("    %s=%ld\n", tensorrt_max_images_var, max_images);

	printf("\n\n");


	// print command line arguments
	printf("[imagenet-console]  args (%i):", argc);

	for( int i=0; i < argc; i++ )
		printf("\n    [%i] %s", i, argv[i]);

	printf("\n\n");

	// create imageNet
	imageNet* net = imageNet::Create(
				caffe_model_val,
				caffe_weights_val,
				imagenet_mean_bin_val,
				imagenet_synset_words_txt_val
				);
	net->EnableProfiler();

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}

	// get image path
	const char* imgPath = NULL;
	char* imagenet_val_path = NULL;
	if( argc == 2 )
	{
		imgPath = argv[1];
	}
	else if( argc == 1 )
	{
		const char* imagenet_val_file = "ILSVRC2012_val_00002212.JPEG"; // 00002212 with AlexNet: top1="no", top5="yes"
		imagenet_val_path = (char*) malloc(strlen(imagenet_val_dir_val) + strlen(imagenet_val_file) + 2);
		sprintf(imagenet_val_path, "%s/%s", imagenet_val_dir_val, imagenet_val_file);
		imgPath = imagenet_val_path;
	}
	else
	{
		printf("Usage: %s [image path]", argv[0]);
		printf(" (by default, a random file from \'%s\')\n", imagenet_val_dir_val);
		return 0;
	}
	printf("[imagenet-console]  image path: \'%s\'\n", imgPath);

	const int rv = classifyImageRGBA(net, imgPath);
	if (imagenet_val_path) { free(imagenet_val_path); }
	
	printf("\nshutting down...\n");
	delete net;
	return rv;
}
