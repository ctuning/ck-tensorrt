/*
 * Loosely based on an TensorRT 1.0 (GIE)
 * sample provided by NVIDIA on Tegra:
 * /usr/src/gie_samples/samples/sampleGoogleNet
 *
 * Therefore, assuming:
 * 2016 (c) NVIDIA
 * 2017 (c) dividiti
 */

#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "tensorNet.h"


#if (1 == CK_TENSORRT_ENABLE_CJSON)
#include <cJSON.h>
#endif

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
#include <xopenme.h>
#endif

#include "NvInfer.h"
#include "NvCaffeParser.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;


#define CHECK(status)                              \
{                                                  \
    if (status != 0)                               \
    {                                              \
        std::cout << "CUDA failure: " << status;   \
        abort();                                   \
    }                                              \
}


// Custom logger for TensorRT info/warning/errors.
class Logger : public ILogger
{
private:
    std::ofstream info_log;

public:
    Logger(const char * path)
    {
        info_log.open(path);
    }

    ~Logger()
    {
        info_log << std::endl;
        info_log.close();
    }

    void log(Severity severity, const char* msg) override
    {
        info_log << msg << std::endl;
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};


// Custom profiler for TensorRT layers.
class Profiler : public IProfiler
{
private:
    const char * cjson_path;
    unsigned int layer_index;
    float total_time_ms;
#if (1 == CK_TENSORRT_ENABLE_CJSON)
    cJSON * dict;
    cJSON * per_layer_info;
#endif

public:
    Profiler(const char * path) :
        cjson_path(path),
        layer_index(0),
        total_time_ms(0.0f)
    {
#if (1 == CK_TENSORRT_ENABLE_CJSON)
        // Create main dictionary.
        dict = cJSON_CreateObject();
        // Create per layer info list.
        per_layer_info = cJSON_CreateArray();
        cJSON_AddItemToObject(dict, "per_layer_info", per_layer_info);
#endif
    }

    ~Profiler()
    {
        std::cout << "\n[tensorrt-time] Total time: " << total_time_ms << " (ms)\n";
#if (1 == CK_TENSORRT_ENABLE_CJSON)
        const char * dict_serialized = cJSON_Print(dict);
        if (cjson_path)
        {
            // Save to file (automatically opened and closed).
            std::ofstream cjson_file(cjson_path);
            cjson_file << dict_serialized;
        }
        else
        {
            // Print to stderr.
            std::cerr << dict_serialized;
        }
        // Deallocate dict.
        cJSON_Delete(dict);
#endif
    }

    virtual void reportLayerTime(const char* layer_name, float layer_ms)
    {
        this->total_time_ms += layer_ms;
        const unsigned int layer_index = this->layer_index++;
        std::cout << "[tensorrt-time]";
        std::cout << " index: " << layer_index;
        std::cout << "; name: " << layer_name;
        std::cout << "; time: " << layer_ms << " (ms)\n";
#if (1 == CK_TENSORRT_ENABLE_CJSON)
        cJSON * new_layer_info = cJSON_CreateObject();
        cJSON * name = cJSON_CreateString(layer_name);
        cJSON * index = cJSON_CreateNumber(layer_index);
        cJSON * time_ms = cJSON_CreateNumber(layer_ms);
        cJSON_AddItemToObject(new_layer_info, "name", name);
        cJSON_AddItemToObject(new_layer_info, "index", index);
        cJSON_AddItemToObject(new_layer_info, "time_ms", time_ms);
        cJSON_AddItemToArray(per_layer_info, new_layer_info);
#endif
    }

};


void convertCaffeToTensorRT(
    const char * deploy_file,                // path to deploy.prototxt file
    const char * weights_file,               // path to caffemodel file
    const std::vector<std::string>& outputs, // network outputs
    size_t max_batch_size,                   // batch size - NB must be at least as large as the batch we want to run with
    bool enable_fp_16,                       // if true and natively supported, use 16-bit floating-point
    std::ostream& output_stream,             // where to serialize the converted model
    Logger& logger                           // custom logger
){
    // Create API root class - must span the lifetime of the engine usage.
    IBuilder* builder = createInferBuilder(logger);
    if (!builder)
    {
        std::cout << "\n[tensorrt-time] Failed to create inference builder (API root class)!\n";
        exit(EXIT_FAILURE);
    }

    builder->setMaxWorkspaceSize(20 << 20);
    // Create network definition.
    INetworkDefinition* network = builder->createNetwork();
    if (!network)
    {
        std::cout << "\n[tensorrt-time] Failed to create network definition!\n";
        exit(EXIT_FAILURE);
    }

    // Parse the Caffe model to populate the network, then set the outputs.
    ICaffeParser* parser = createCaffeParser();
    if (!parser)
    {
        std::cout << "\n[tensorrt-time] Failed to create Caffe parser!\n";
        exit(EXIT_FAILURE);
    }

    // Check whether 16-bit floating-point is natively supported.
    const bool has_fp_16 = builder->platformHasFastFp16();
    // Create a 16-bit model if supported and enabled.
    const bool use_fp_16 = has_fp_16 && enable_fp_16;
    DataType data_type = use_fp_16 ? DataType::kHALF : DataType::kFLOAT;

    // The third parameter is the network definition that the parser will populate.
    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(deploy_file, weights_file, *network, data_type);
    if (!blobNameToTensor)
    {
        std::cout << "\n[tensorrt-time] Failed to parse Caffe model!\n";
        exit(EXIT_FAILURE);
    }

    // As the Caffe model has no notion of outputs, we need to specify
    // explicitly which tensors the engine should generate.
    for (auto& output : outputs)
    {
        const char * output_name = output.c_str();
        ITensor* tensor = blobNameToTensor->find(output_name);
        if (!tensor)
        {
            std::cerr << "\n[tensorrt-time] Failed to retrieve tensor for output \'" << output_name << "\'!\n";
            exit(EXIT_FAILURE);
        }
        network->markOutput(*tensor);
    }

    // Build the engine.
    builder->setMaxBatchSize(max_batch_size);

    // Set up the network for paired-fp16 format if supported and enabled.
    builder->setHalf2Mode(use_fp_16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
    {
        std::cerr << "\n[tensorrt-time] Failed to build CUDA engine!\n";
        exit(EXIT_FAILURE);
    }

    // We no longer need the network, nor do we need the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then shut everything down.
#if NV_TENSORRT_MAJOR > 1
	  nvinfer1::IHostMemory* serMem = engine->serialize();
	  if( !serMem )
	  {
        std::cerr << "\n[tensorrt-time] failed to serialize CUDA engine!\n";
        exit(EXIT_FAILURE);
	  }
    output_stream.write((const char*)serMem->data(), serMem->size());
#else
    engine->serialize(output_stream);
#endif

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void timeInference(
    ICudaEngine* engine,
    const size_t tensorrt_batch_size,
    const char * tensorrt_input_blob_name,
    const char * tensorrt_output_blob_name,
    Profiler& profiler
){
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    // FIXME: "we know .. exactly one input and one output" - for GoogleNet, AlexNet?
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than ICudaEngine::getNbBindings().
    const int inputIndex  = engine->getBindingIndex(tensorrt_input_blob_name);
    const int outputIndex = engine->getBindingIndex(tensorrt_output_blob_name);

    // Allocate GPU buffers.
    auto inputDims = engine->getBindingDimensions(inputIndex);
    auto outputDims = engine->getBindingDimensions(outputIndex);
    const size_t inputSize = tensorrt_batch_size * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
    const size_t outputSize = tensorrt_batch_size * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "\n[tensorrt-time] Failed to create execution context!\n";
        exit(EXIT_FAILURE);
    }

    // Set the custom profiler.
    context->setProfiler(&profiler);

    // Zero the input buffer.
    CHECK(cudaMemset(buffers[inputIndex], 0, inputSize));

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
    xopenme_clock_start(0);
#endif

    context->execute(tensorrt_batch_size, buffers);

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
    xopenme_clock_end(0);
#endif

    // Release the context and buffers.
    context->destroy();
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
    int exit_status = EXIT_SUCCESS;

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
     xopenme_init(1, 0);
#endif

    // Print environment variables set by CK.
    printf("\n[tensorrt-time] CK settings detected:\n");

    const char * caffe_model_var = "CK_CAFFE_MODEL";
    const char * caffe_model_val = getenv(caffe_model_var);
    printf("     %s=\"%s\"\n", caffe_model_var,
                               caffe_model_val ? caffe_model_val : "?");

    const char * caffe_weights_var = "CK_ENV_MODEL_CAFFE_WEIGHTS";
    const char * caffe_weights_val = getenv(caffe_weights_var);
    printf("     %s=\"%s\"\n", caffe_weights_var,
                               caffe_weights_val ? caffe_weights_val : "?");

    // TODO (elsewhere): Augment the Caffe model packages with the input and output blob names.
    const char * caffe_model_input_blob_name_var = "CK_CAFFE_MODEL_INPUT_BLOB_NAME";
    const char * caffe_model_input_blob_name_val = getenv(caffe_model_input_blob_name_var);
    printf("     %s=\"%s\"\n", caffe_model_input_blob_name_var,
                               caffe_model_input_blob_name_val ? caffe_model_input_blob_name_val : "?");

    const char * caffe_model_output_blob_name_var = "CK_CAFFE_MODEL_OUTPUT_BLOB_NAME";
    const char * caffe_model_output_blob_name_val = getenv(caffe_model_output_blob_name_var);
    printf("     %s=\"%s\"\n", caffe_model_output_blob_name_var,
                               caffe_model_output_blob_name_val ? caffe_model_output_blob_name_val : "?");

    const char * caffe_batch_size_var = "CK_CAFFE_BATCH_SIZE";
    const char * caffe_batch_size_val = getenv(caffe_batch_size_var);
    printf("     %s=\"%s\"\n", caffe_batch_size_var,
                               caffe_batch_size_val ? caffe_batch_size_val : "?");

    const char * tensorrt_enable_fp16_var = "CK_TENSORRT_ENABLE_FP16";
    const char * tensorrt_enable_fp16_val = getenv(tensorrt_enable_fp16_var);
    printf("     %s=\"%s\"\n", tensorrt_enable_fp16_var,
                               tensorrt_enable_fp16_val ? tensorrt_enable_fp16_val : "?");

    const char * tensorrt_enable_cache_var = "CK_TENSORRT_ENABLE_CACHE";
    const char * tensorrt_enable_cache_val = getenv(tensorrt_enable_cache_var);
    printf("     %s=\"%s\"\n", tensorrt_enable_cache_var,
                               tensorrt_enable_cache_val ? tensorrt_enable_cache_val : "?");

    const char * tensorrt_cjson_path_var = "CK_TENSORRT_CJSON_PATH";
    const char * tensorrt_cjson_path_val = getenv(tensorrt_cjson_path_var);
    printf("     %s=\"%s\"\n", tensorrt_cjson_path_var,
                               tensorrt_cjson_path_val ? tensorrt_cjson_path_val : "?");
    Profiler profiler(tensorrt_cjson_path_val);

    const char * tensorrt_info_log_var = "CK_TENSORRT_INFO_LOG";
    const char * tensorrt_info_log_val = getenv(tensorrt_info_log_var);
    printf("     %s=\"%s\"\n", tensorrt_info_log_var,
                               tensorrt_info_log_val ? tensorrt_info_log_val : "?");
    Logger logger(tensorrt_info_log_val);

    // Print configuration variables inferred.
    printf("\n[tensorrt-time] TensorRT settings inferred:\n");
    const char * tensorrt_input_blob_name = caffe_model_input_blob_name_val ? caffe_model_input_blob_name_val : "data";
    printf("     TENSORRT_INPUT_BLOB_NAME=\"%s\"\n", tensorrt_input_blob_name);

    const char * tensorrt_output_blob_name = caffe_model_output_blob_name_val ? caffe_model_output_blob_name_val : "prob";
    printf("     TENSORRT_OUTPUT_BLOB_NAME=\"%s\"\n", tensorrt_output_blob_name);

    const size_t tensorrt_batch_size = caffe_batch_size_val ? atoi(caffe_batch_size_val) : 1;
    printf("     TENSORRT_BATCH_SIZE=%ld\n", tensorrt_batch_size);

    const bool   tensorrt_enable_fp16 = tensorrt_enable_fp16_val ? (bool)atoi(tensorrt_enable_fp16_val) : true;
    printf("     TENSORRT_ENABLE_FP16=%d\n", tensorrt_enable_fp16);

    const bool   tensorrt_enable_cache = tensorrt_enable_cache_val ? (bool)atoi(tensorrt_enable_cache_val) : true;
    printf("     TENSORRT_ENABLE_CACHE=%d\n", tensorrt_enable_cache);

    // Print the basic engine info.
    std::cout << "\n[tensorrt-time] Starting a TensorRT engine:";
    std::cout << "\n- for the Caffe model at \'" << caffe_weights_val << "\'";
    std::cout << "\n- with the batch size of " << tensorrt_batch_size;
    std::cout << "\n- with 16-bit floating point " << (tensorrt_enable_fp16 ? "enabled" : "disabled");
    std::cout << "\n- with model caching " << (tensorrt_enable_cache ? "enabled" : "disabled");
    std::cout << std::endl;

    // Parse the Caffe model or load from a cache.
    std::stringstream tensorrt_model_stream;
    tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);
    if (!tensorrt_enable_cache)
    {
        std::cout << "\n[tensorrt-time] Converting the Caffe model into a TensorRT one...\n";
        std::vector<std::string> tensorrt_model_outputs({tensorrt_output_blob_name});
        convertCaffeToTensorRT(caffe_model_val, caffe_weights_val,
            tensorrt_model_outputs, tensorrt_batch_size, tensorrt_enable_fp16, tensorrt_model_stream,
            logger);
    }
    else
    {
        // Look up using the following "cache tag":
        // "<caffe weights file>.tensorrt-<version>.fp<precision bits>.bs<batch size>".
        const int version = getInferLibVersion();
#if NV_TENSORRT_MAJOR > 1
        const int version_major = version / 1000;
        const int version_minor = (version - version_major * 1000) / 100;
        const int version_patch = version - version_major * 1000 - version_minor * 100;
#else /* NV_TENSORRT_MAJOR == 1 */
        const int version_major = version >> 16;
        const int version_minor = (version & ((1<<16)-1)) >> 8;
        const int version_patch = (version & ((1<<8)-1));
#endif /* NV_TENSORRT_MAJOR */
        std::cout << "\n[tensorrt-time] getInferLibVersion(): " << version;
        std::cout << "\n[tensorrt-time] TensorRT version: " << version_major << "." << version_minor << "." << version_patch << "\n";

        std::stringstream tensorrt_model_cache_ss;
        tensorrt_model_cache_ss << caffe_weights_val;
        tensorrt_model_cache_ss << ".tensorrt-" << version_major << "." << version_minor << "." << version_patch;
        tensorrt_model_cache_ss << ".fp" << (tensorrt_enable_fp16 ? "16" : "32");
#if NV_TENSORRT_MAJOR < 3
        tensorrt_model_cache_ss << ".bs" << tensorrt_batch_size;
#endif
        // Try to load the file.
        const std::string tensorrt_model_cache_path(tensorrt_model_cache_ss.str());
        std::cout << "\n[tensorrt-time] Checking if cached at \'" << tensorrt_model_cache_path << "\'...";
        std::ifstream tensorrt_model_cache_load(tensorrt_model_cache_path);
        if (tensorrt_model_cache_load)
        {
            std::cout << "\n[tensorrt-time] - cached TensorRT model found, loading...";
            tensorrt_model_stream << tensorrt_model_cache_load.rdbuf();
            tensorrt_model_cache_load.close();
            std::cout << "\n[tensorrt-time] - cached TensorRT model loaded.";
        }
        else
        {
            std::cout << "\n[tensorrt-time] - cached TensorRT model not found, converting...";
            std::vector<std::string> tensorrt_model_outputs({tensorrt_output_blob_name});
            convertCaffeToTensorRT(caffe_model_val, caffe_weights_val,
                tensorrt_model_outputs, tensorrt_batch_size, tensorrt_enable_fp16, tensorrt_model_stream,
                logger);
            std::cout << "\n[tensorrt-time] - caching TensorRT model...";
            std::ofstream tensorrt_model_cache_store(tensorrt_model_cache_path);

            if (! tensorrt_model_cache_store) {
                std::cerr << "\n[tensorrt-time] Failed to cache TensorRT model!\n";
                exit(EXIT_FAILURE);
            }

            tensorrt_model_cache_store << tensorrt_model_stream.rdbuf();
            tensorrt_model_cache_store.close();
            std::cout << "\n[tensorrt-time] - TensorRT model cached.";
        }
        tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);
    }

    // Create inference runtime engine.
    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "\n[tensorrt-time] Failed to create inference runtime!\n";
        exit(EXIT_FAILURE);
    }

#if NV_TENSORRT_MAJOR > 1
    std::cout << "\n[tensorrt-time] Running with NV_TENSORRT_MAJOR > 1\n";
    // support for stringstream deserialization was deprecated in TensorRT v2
    // instead, read the stringstream into a memory buffer and pass that to TRT.
    tensorrt_model_stream.seekg(0, std::ios::end);
    const int modelSize = tensorrt_model_stream.tellg();
    tensorrt_model_stream.seekg(0, std::ios::beg);

    void* modelMem = malloc(modelSize);
    if( !modelMem )
    {
        std::cerr << "\n[tensorrt-time] Failed to allocate memory to deserialize model!\n";
        exit(EXIT_FAILURE);
    }

    tensorrt_model_stream.read((char*)modelMem, modelSize);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
    free(modelMem);
#else
    std::cout << "\n[tensorrt-time] Running with NV_TENSORRT_MAJOR == 1\n";

	  // TensorRT v1 can deserialize directly from stringstream
	  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(tensorrt_model_stream);
#endif

    if (!engine)
    {
        std::cerr << "\n[tensorrt-time] Failed to deserialize CUDA engine!\n";
        exit(EXIT_FAILURE);
    }

    // Run inference with zero data to measure performance.
    std::cout << "\n[tensorrt-time] Running inference...\n";
    timeInference(engine,
        tensorrt_batch_size, tensorrt_input_blob_name, tensorrt_output_blob_name,
        profiler);

    std::cout << "\n[tensorrt-time] Shutting down...\n";
    engine->destroy();
    runtime->destroy();

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
     xopenme_dump_state();
     xopenme_finish();
#endif

    return exit_status;
}
