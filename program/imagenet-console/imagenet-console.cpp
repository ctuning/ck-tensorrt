/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"



// main entry point
int main( int argc, char** argv )
{
	// print environment variables set by CK
	printf("imagenet-console\n  ck-env:\n");

	const char * caffe_model = "CK_CAFFE_MODEL";
	const char * caffe_model_value = getenv(caffe_model);
	printf("    %s=\"%s\"\n", caffe_model, caffe_model_value ? caffe_model_value : "?");

	const char * caffe_weights = "CK_ENV_MODEL_CAFFE_WEIGHTS";
	const char * caffe_weights_value = getenv(caffe_weights);
	printf("    %s=\"%s\"\n", caffe_weights, caffe_weights_value ? caffe_weights_value : "?");

	const char * imagenet_mean_bin = "CK_CAFFE_IMAGENET_MEAN_BIN";
	const char * imagenet_mean_bin_value = getenv(imagenet_mean_bin);
	printf("    %s=\"%s\"\n", imagenet_mean_bin, imagenet_mean_bin_value ? imagenet_mean_bin_value : "?");

	const char * imagenet_synset_words_txt = "CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT";
	const char * imagenet_synset_words_txt_value = getenv(imagenet_synset_words_txt);
	printf("    %s=\"%s\"\n", imagenet_synset_words_txt, imagenet_synset_words_txt_value ? imagenet_synset_words_txt_value : "?");

	const char * caffe_iterations = "CK_CAFFE_ITERATIONS";
	const char * caffe_iterations_value = getenv(caffe_iterations);
	printf("    %s=%s\n", caffe_iterations, caffe_iterations_value ? caffe_iterations_value : "?");

	const char * caffe_batch_size = "CK_CAFFE_BATCH_SIZE";
	const char * caffe_batch_size_value = getenv(caffe_batch_size);
	printf("    %s=%s\n", caffe_batch_size, caffe_batch_size_value ? caffe_batch_size_value : "?");

	printf("\n\n");


	// print command line arguments
	printf("imagenet-console\n  args (%i):", argc);

	for( int i=0; i < argc; i++ )
		printf("\n    [%i] %s", i, argv[i]);

	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc < 2 )
	{
		printf("imagenet-console:   input image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	

	// create imageNet
	imageNet* net = imageNet::Create(
				caffe_model_value,
				caffe_weights_value,
				imagenet_mean_bin_value,
				imagenet_synset_words_txt_value
				);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	float confidence = 0.0f;
	
	// classify image
	const int img_class = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);
	
	if( img_class < 0 )
	{
		printf("imagenet-console:  failed to classify '%s'  (result=%i)\n", imgFilename, img_class);
	}
	else
	{
		printf("imagenet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, net->GetClassDesc(img_class));
	}
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
