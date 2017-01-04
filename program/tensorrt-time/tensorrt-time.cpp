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
#include <fstream>
#include <sstream>
#include <iostream>
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
    cJSON * dict;
    cJSON * per_layer_info;
    unsigned int index;

public:
    Profiler()
    {
        // Create main dictionary.
        dict = cJSON_CreateObject();
        // Create per layer info list.
        per_layer_info = cJSON_CreateArray();
        cJSON_AddItemToObject(dict, "per_layer_info", per_layer_info);
        // Init layer index.
        index = 0;
    }

    ~Profiler()
    {
        // Print dict to stderr. TODO: Save directly to file.
        const char * dict_serialized = cJSON_Print(dict);
        std::cerr << dict_serialized << std::endl;
        // Deallocate dict.
        cJSON_Delete(dict);
    }

    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;

#if (1 == CK_TENSORRT_ENABLE_CJSON)
        cJSON * new_layer_info = cJSON_CreateObject();
        cJSON * name = cJSON_CreateString(layerName);
        cJSON * index = cJSON_CreateNumber(this->index++);
        cJSON * time_ms = cJSON_CreateNumber(ms);
        cJSON_AddItemToObject(new_layer_info, "name", name);
        cJSON_AddItemToObject(new_layer_info, "index", index);
        cJSON_AddItemToObject(new_layer_info, "time_ms", time_ms);
        cJSON_AddItemToArray(per_layer_info, new_layer_info);
#endif
    } // reportLayerTime()

    void printLayerTimes()
    {
        float totalTime = 0.0f;
        for (size_t i = 0; i < mProfile.size(); ++i)
        {
            printf("%-40.40s %4.3f ms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Total time: %4.3f\n", totalTime / TIMING_ITERATIONS);

#if (1 == CK_TENSORRT_ENABLE_CJSON)

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
    INetworkDefinition* network = builder->createNetwork();

    // Parse the Caffe model to populate the network, then set the outputs.
    ICaffeParser* parser = createCaffeParser();

    // Check whether 16-bit floating-point is natively supported.
    const bool hasFp16 = builder->platformHasFastFp16();
    // Create a 16-bit model if supported and enabled.
    const bool useFp16 = hasFp16 && enableFp16;
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;

    // The third parameter is the network definition that the parser will populate.
    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(modelFile, modelWeightsFile, *network, modelDataType);

    assert(blobNameToTensor != nullptr);
    // As the Caffe model has no notion of outputs, we need to specify
    // explicitly which tensors the engine should generate.
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    // Set up the network for paired-fp16 format if supported and enabled.
    if (useFp16) builder->setHalf2Mode(true);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

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
    printf("\n[tensorrt-time]  CK settings detected:\n");

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

    // Print configuration variables inferred.
    printf("\n[tensorrt-time]  TensorRT settings inferred:\n");
    const char * tensorrt_input_blob_name = caffe_model_input_blob_name_val ? caffe_model_input_blob_name_val : "data";
    printf("     TENSORRT_INPUT_BLOB_NAME=\"%s\"\n", tensorrt_input_blob_name);

    const char * tensorrt_output_blob_name = caffe_model_output_blob_name_val ? caffe_model_output_blob_name_val : "prob";
    printf("     TENSORRT_OUTPUT_BLOB_NAME=\"%s\"\n", tensorrt_output_blob_name);

    const size_t tensorrt_batch_size = caffe_batch_size_val ? atoi(caffe_batch_size_val) : 1;
    printf("     TENSORRT_BATCH_SIZE=%ld\n", tensorrt_batch_size);

    const bool   tensorrt_enable_fp16 = tensorrt_enable_fp16_val ? (bool)atoi(tensorrt_enable_fp16_val) : true;
    printf("     TENSORRT_ENABLE_FP16=%d\n", tensorrt_enable_fp16);

    // Print the basic engine info.
    std::cout << "\n[tensorrt-time]  Building and running a TensorRT engine";
    std::cout << "\nfor \'" << caffe_weights_val << "\'";
    std::cout << "\nwith the batch size of " << tensorrt_batch_size;
    std::cout << " and 16-bit floating point " << (tensorrt_enable_fp16 ? "enabled" : "disabled") << std::endl;

    // Parse the Caffe model and the mean file. FIXME: Only for AlexNet?
    std::stringstream tensorrt_model_stream;
    tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);
    std::vector<std::string> tensorrt_model_outputs({tensorrt_output_blob_name});
    caffeToGIEModel(caffe_model_val, caffe_weights_val,
                    tensorrt_model_outputs, tensorrt_batch_size, tensorrt_enable_fp16,
                    tensorrt_model_stream);

    // Create an engine.
    IRuntime* infer = createInferRuntime(gLogger);
    ICudaEngine* engine = infer->deserializeCudaEngine(tensorrt_model_stream);

    // Run inference with zero data to measure performance.
    timeInference(engine, tensorrt_batch_size, tensorrt_input_blob_name, tensorrt_output_blob_name);

    std::cout << "\n[tensorrt-time]  Printing per layer timing info...\n";
    gProfiler.printLayerTimes();

    std::cout << "\n[tensorrt-time]  Shutting down...\n";
    engine->destroy();
    infer->destroy();

#if (1 == CK_TENSORRT_ENABLE_XOPENME)
     xopenme_dump_state();
     xopenme_finish();
#endif

    return exit_status;
}
