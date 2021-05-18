#include "yoloPlugin.h"
#include <assert.h>
#include <iostream>
#include <common.h>
using namespace nvinfer1;
using nvinfer1::Yolo;
using nvinfer1::YoloPluginCreator;
//std::vector<TensorInfo> m_plugin_OutputTensors;
namespace
{
    //const char* Yolo_PLUGIN_VERSION{ "1" };
    //const char* Yolo_PLUGIN_NAME{ "YOLO" };
    // Write values into buffer
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    // Read values from buffer
    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }
} // namespace

namespace nvinfer1
{
    Yolo::Yolo()
    {

    }

    Yolo::Yolo(int numclass, int gride_w, int gride_h, int numanchors)
    {
        numanchors_ = numanchors;
        numclass_ = numclass;
        _n_grid_h = gride_w;
        _n_grid_w = gride_h;
    }

    Yolo::Yolo(const void* buffer, size_t length)
    {
        const char* d = reinterpret_cast<const char*>(buffer), * a = d;
        numanchors_ = read<int>(d);
        numclass_ = read<int>(d);
        _n_grid_h = read<int>(d);
        _n_grid_w = read<int>(d);
        assert(d == a + length);
    }

    Yolo::~Yolo()
    {

    }

    int Yolo::getNbOutputs() const
    {
        return 1;
    }

    Dims Yolo::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(inputs[0].nbDims == 3);
        return inputs[0];
    }

    int Yolo::initialize()
    {
        return 0;
    }

    void Yolo::terminate()
    {

    }

    size_t Yolo::getWorkspaceSize(int maxBatchSize) const
    {
        return 0;
    }


    //new cuda
    inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

    __global__ void gpuYoloLayerV3(const float* input, float* output, const uint32_t grid_h_,
        const uint32_t grid_w_, const uint32_t numOutputClasses,
        const uint32_t numBBoxes)
    {
        uint32_t x_id = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y_id = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t z_id = blockIdx.z * blockDim.z + threadIdx.z;

        if ((x_id >= grid_w_) || (y_id >= grid_h_) || (z_id >= numBBoxes))
        {
            return;
        }

        const int numGridCells = grid_h_ * grid_w_;
        const int bbindex = y_id * grid_w_ + x_id;

        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
            = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
            = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

        for (uint32_t i = 0; i < numOutputClasses; ++i)
        {
            output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
                = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
        }
    }

    void cudaYoloLayerV3(const void* input, void* output, const uint32_t& batchSize,
        const uint32_t& n_grid_h_, const uint32_t& n_grid_w_,
        const uint32_t& numOutputClasses, const uint32_t& numBBoxes,
        uint64_t outputSize, cudaStream_t stream)
    {
        dim3 threads_per_block(16, 16, 4);
        dim3 number_of_blocks((n_grid_w_ / threads_per_block.x) + 1,
            (n_grid_h_ / threads_per_block.y) + 1,
            (numBBoxes / threads_per_block.z) + 1);
        for (int batch = 0; batch < batchSize; ++batch)
        {
            gpuYoloLayerV3 << <number_of_blocks, threads_per_block, 0, stream >> > (
                reinterpret_cast<const float*>(input) + (batch * outputSize),
                reinterpret_cast<float*>(output) + (batch * outputSize), n_grid_h_, n_grid_w_, numOutputClasses,
                numBBoxes);
        }
        //return cudaGetLastError();
    }

    int Yolo::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        cudaYoloLayerV3(inputs[0], outputs[0], batchSize, _n_grid_h, _n_grid_w, numclass_, numanchors_, _n_grid_h * _n_grid_w * numanchors_ * (5 + numclass_), stream);

        return 0;
    }

    size_t Yolo::getSerializationSize() const
    {
        return sizeof(numanchors_) + sizeof(numclass_) + sizeof(_n_grid_h) + sizeof(_n_grid_w);
    }
    //jie shou zou bo de cheng guo.
    void Yolo::serialize(void* buffer) const
    {
        char* d = static_cast<char*>(buffer), * a = d;
        write(d, numanchors_);
        write(d, numclass_);
        write(d, _n_grid_h);
        write(d, _n_grid_w);
        assert(d == a + getSerializationSize());
    }

    // Set plugin namespace
    void Yolo::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* Yolo::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType Yolo::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        assert(index == 0);
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool Yolo::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool Yolo::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void Yolo::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void Yolo::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void Yolo::detachFromContext() {}

    const char* Yolo::getPluginType() const
    {
        return "YOLO";
    }

    const char* Yolo::getPluginVersion() const
    {
        return "1";
    }

    void Yolo::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* Yolo::clone() const
    {
        // Create a new instance
        Yolo* plugin = new Yolo(numclass_, _n_grid_w, _n_grid_h, numanchors_);
        // Set the namespace
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {

        mPluginAttributes.emplace_back(PluginField("classes", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("gride_w", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("gride_h", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("anchor_num", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const
    {
        return "YOLO";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        //assert(!strcmp(name, getPluginName()));
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "classes"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                numclass_ = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "gride_w")) {
                assert(fields[i].type == PluginFieldType::kINT32);
                _n_grid_w = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "gride_h")) {
                assert(fields[i].type == PluginFieldType::kINT32);
                _n_grid_h = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "anchor_num")) {
                assert(fields[i].type == PluginFieldType::kINT32);
                numanchors_ = *(static_cast<const int*>(fields[i].data));
            }
        }
        Yolo* obj = new Yolo(numclass_, _n_grid_w, _n_grid_h, numanchors_);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        Yolo* obj = new Yolo(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}