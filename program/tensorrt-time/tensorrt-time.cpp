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


// NB: Using multiple timing iterations takes away the ability
// to assess the minimum execution time, performance variability, etc.
static const int TIMING_ITERATIONS = CK_TENSORRT_ITERATIONS;


// Logger for GIE info/warning/errors.
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if (severity!=Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


// Profiler for GIE layers.
class Profiler : public IProfiler
{
private:
    unsigned int layer_index;
    float total_time_ms;
#if (1 == CK_TENSORRT_ENABLE_CJSON)
    cJSON * dict;
    cJSON * per_layer_info;
#endif

public:
    Profiler() : layer_index(0), total_time_ms(0.0f)
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
        // Print dict to stderr. TODO: Save directly to file.
        const char * dict_serialized = cJSON_Print(dict);
        std::cerr << dict_serialized << std::endl;
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

} gProfiler;


void caffeToGIEModel(const char * modelFile,                    // path to deploy.prototxt file
                     const char * modelWeightsFile,             // path to caffemodel file
                     const std::vector<std::string>& outputs,   // network outputs
                     size_t maxBatchSize,                       // batch size - NB must be at least as large as the batch we want to run with
                     bool enableFp16,                           // if true and natively supported, use 16-bit floating-point
                     std::ostream& gieModelStream)
{
    // Create API root class - must span the lifetime of the engine usage.
    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder)
    {
        std::cout << "\n[tensorrt-time] Failed to create inference builder (API root class)!\n";
        exit(EXIT_FAILURE);
    }

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
    const bool hasFp16 = builder->platformHasFastFp16();
    // Create a 16-bit model if supported and enabled.
    const bool useFp16 = hasFp16 && enableFp16;
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;

    // The third parameter is the network definition that the parser will populate.
    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(modelFile, modelWeightsFile, *network, modelDataType);
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
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    // Set up the network for paired-fp16 format if supported and enabled.
    builder->setHalf2Mode(useFp16);

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
    engine->serialize(gieModelStream);
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void timeInference(ICudaEngine* engine,
                   const size_t tensorrt_batch_size,
                   const char * tensorrt_input_blob_name,
                   const char * tensorrt_output_blob_name)
{
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
    Dims3 inputDims = engine->getBindingDimensions(inputIndex);
    Dims3 outputDims = engine->getBindingDimensions(outputIndex);
    const size_t inputSize = tensorrt_batch_size * inputDims.c * inputDims.h * inputDims.w * sizeof(float);
    const size_t outputSize = tensorrt_batch_size * outputDims.c * outputDims.h * outputDims.w * sizeof(float);

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "\n[tensorrt-time] Failed to create execution context!\n";
        exit(EXIT_FAILURE);
    }

    // Set the customized profiler.
    context->setProfiler(&gProfiler);

    // Zero the input buffer.
    CHECK(cudaMemset(buffers[inputIndex], 0, inputSize));

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
    xopenme_clock_start(0);
#endif

    for (int i = 0; i < TIMING_ITERATIONS; i++)
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
        std::cout << "\n[tensorrt-time] Converting the Caffe model to a TensorRT one...";
        std::vector<std::string> tensorrt_model_outputs({tensorrt_output_blob_name});
        caffeToGIEModel(caffe_model_val, caffe_weights_val,
                        tensorrt_model_outputs, tensorrt_batch_size, tensorrt_enable_fp16,
                        tensorrt_model_stream);
    }
    else
    {
        // Look up using the following "cache tag":
        // "<caffe weights file>.tensorrt-<version>.fp<precision bits>.bs<batch size>".
        const int version = getInferLibVersion();
        const int version_major = version >> 16;
        const int version_minor = (version & ((1<<16)-1)) >> 8;
        const int version_patch = (version & ((1<<8)-1));

        std::stringstream tensorrt_model_cache_ss;
        tensorrt_model_cache_ss << caffe_weights_val;
        tensorrt_model_cache_ss << ".tensorrt-" << version_major << "." << version_minor << "." << version_patch;
        tensorrt_model_cache_ss << ".fp" << (tensorrt_enable_fp16 ? "16" : "32");
        tensorrt_model_cache_ss << ".bs" << (tensorrt_batch_size);

        // Try to load the file.
        const std::string tensorrt_model_cache_path(tensorrt_model_cache_ss.str());
        std::cout << "\n[tensorrt-time] Checking if cached at \'" << tensorrt_model_cache_path << "\'...";
        std::ifstream tensorrt_model_cache_load(tensorrt_model_cache_path);
        if (tensorrt_model_cache_load)
        {
            std::cout << "\n[tensorrt-time] - found, loading...";
            tensorrt_model_stream << tensorrt_model_cache_load.rdbuf();
            tensorrt_model_cache_load.close();
        }
        else
        {
            std::cout << "\n[tensorrt-time] - not found, converting...";
            std::vector<std::string> tensorrt_model_outputs({tensorrt_output_blob_name});
            caffeToGIEModel(caffe_model_val, caffe_weights_val,
                            tensorrt_model_outputs, tensorrt_batch_size, tensorrt_enable_fp16,
                            tensorrt_model_stream);
            std::cout << "\n[tensorrt-time] - storing...";
            std::ofstream tensorrt_model_cache_store(tensorrt_model_cache_path);
            tensorrt_model_cache_store << tensorrt_model_stream.rdbuf();
            tensorrt_model_cache_store.close();
        }
        tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);
    }

    // Create inference runtime engine.
    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime)
    {
        std::cerr << "\n[tensorrt-time] Failed to create inference runtime!\n";
        exit(EXIT_FAILURE);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(tensorrt_model_stream);
    if (!engine)
    {
        std::cerr << "\n[tensorrt-time] Failed to deserialize CUDA engine!\n";
        exit(EXIT_FAILURE);
    }

    // Run inference with zero data to measure performance.
    std::cout << "\n[tensorrt-time] Running inference...\n";
    timeInference(engine, tensorrt_batch_size, tensorrt_input_blob_name, tensorrt_output_blob_name);

    std::cout << "\n[tensorrt-time] Shutting down...\n";
    engine->destroy();
    runtime->destroy();

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
     xopenme_dump_state();
     xopenme_finish();
#endif

    return exit_status;
}
