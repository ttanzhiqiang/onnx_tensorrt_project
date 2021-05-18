/*
 * custom plugin, TensorRT-7
 *
 */ 
#ifndef DCN_V2_HPP
#define DCN_V2_HPP

#include "NvInfer.h"
#include "common.h"
#include <thread>
#include <cassert>
#include <iostream>
#include <vector>

constexpr const char* DCN_PLUGIN_VERSION{"1"};
constexpr const char* DCN_PLUGIN_NAME{"DCNv2"};

namespace
{
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

/// inherited from IPluginV2DynamicExt 
class DCNDynamicPlugin final: public nvinfer1::IPluginV2DynamicExt {
private:
    std::string _nameSpace;
    nvinfer1::Dims _inputDims;
    nvinfer1::Dims _outputDims;
    int _kernel_size;
    int _dilation;
    int _deformable_groups;
    int _padding;
    int _stride;
    bool _initialized;
    float* _d_ones;
    float* _d_columns;
    float* _d_weight;

public:
    DCNDynamicPlugin(const void* data, size_t length) {
        //using namespace SeriBuff;
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        _inputDims.nbDims = 3;
        _inputDims = nvinfer1::Dims3();
        _inputDims.d[0] = read<int>(d);
        _inputDims.d[1] = read<int>(d);
        _inputDims.d[2] = read<int>(d);
        _outputDims.nbDims = 3;
        _outputDims.d[0] = read<int>(d);
        _outputDims.d[1] = read<int>(d);
        _outputDims.d[2] = read<int>(d);
        _kernel_size = read<int>(d);
        _dilation  = read<int>(d);
        _padding   = read<int>(d);
        _stride    = read<int>(d);
        _deformable_groups = read<int>(d);

        assert(d == a + length);
        //std::cout << "**** DCNPlugin Constructor2 has been called!" << std::endl;
        _initialized = false;
         initialize();
    }
    DCNDynamicPlugin(int in_channel,
        int out_channel, int in_width, int in_height, int out_width, int out_height,
        int kernel,
        int deformable_group,
        int dilation,
        int padding,
        int stride ) {
        _inputDims.nbDims = 3;
        _inputDims = nvinfer1::Dims3();
        _inputDims.d[0] = in_channel;
        _inputDims.d[1] = in_height;
        _inputDims.d[2] = in_width;
        _outputDims.nbDims = 3;
        _outputDims.d[0] = out_channel;
        _outputDims.d[1] = out_height;
        _outputDims.d[2] = out_width;
        _kernel_size = kernel;
        _dilation = dilation;
        _padding = padding;
        _stride = stride;
        _deformable_groups = deformable_group;
        _initialized = false;
        initialize();
    }

    ~DCNDynamicPlugin() override {
    }
    DCNDynamicPlugin()=delete; //constructor must has arguments.
    nvinfer1::Dims const&  getInputDims(int index) const { return _inputDims; }

    /// override these methods
    int getNbOutputs() const override {return 1; }
    void terminate() override;
    void destroy() override { delete this; }

    int initialize() override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputDims, int nbInputs) override {
      //std::cout << "**** getOutputDimensions called! **** id:" << this << std::endl;
      assert(index == 0);
      assert(nbInputs == 5);
      auto& input = inputDims[0];
      assert(3 == input.nbDims);  /// CHW
      assert(input.d[0] > 0 && input.d[1] > 0 && input.d[2] > 0 );

      //printf("Inputs & outputs'shape: (%d, %d, %d), (%d, %d, %d)", _inputDims.d[0], _inputDims.d[1], _inputDims.d[2], _outputDims.d[0], _outputDims.d[1], _outputDims.d[2]);
      return _outputDims;
    }

    //int enqueue(int batch_size, const void* const* inputs, \
    //        void** outputs, void* workspace, cudaStream_t stream) override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) override;



    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, \
            int nbInputs) const override {assert(index == 0); 
        //return this->_dataType;
        return nvinfer1::DataType::kFLOAT;
    }

    size_t getSerializationSize() const override {
        return sizeof(int) * 11;
    }
    /// serialize the engine
    void serialize(void* buffer) const override {
        //using namespace SeriBuff;
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, _inputDims.d[0]);
        write(d, _inputDims.d[1]);
        write(d, _inputDims.d[2]);
        write(d, _outputDims.d[0]);
        write(d, _outputDims.d[1]);
        write(d, _outputDims.d[2]);
        write(d, _kernel_size);
        write(d, _dilation);
        write(d, _padding);
        write(d, _stride);
        write(d, _deformable_groups);

        assert(d == a + getSerializationSize());
    }

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new DCNDynamicPlugin(*this);
    }

    /// support format: fp32/fp16 and NCHW
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        //return ((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) 
        return (type == nvinfer1::DataType::kFLOAT  
                && format == nvinfer1::PluginFormat::kNCHW);
    }
    const char* getPluginType() const override {return DCN_PLUGIN_NAME;}
    const char* getPluginVersion() const override {return DCN_PLUGIN_VERSION;}
    void setPluginNamespace(const char* libNamespace) override {_nameSpace = libNamespace;}
    //const char* getPluginNamespace() const {return _nameSpace.c_str();}
    const char* getPluginNamespace() const {return "";}
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override {return false;}
    bool canBroadcastInputAcrossBatch(int inputIndex) const override {return false;}
    void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator
            ) override {}
    void detachFromContext() override {}

    //IPluginV2DynamicExt
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
        return 0;
    }
};

/// IPluginCreator
class DCNDynamicPluginCreator : public nvinfer1::IPluginCreator {
private:
    std::string mNamespace;
    static nvinfer1::PluginFieldCollection _mFC;
    static std::vector<nvinfer1::PluginField> _mPluginAttributes;
public:
    DCNDynamicPluginCreator();
    ~DCNDynamicPluginCreator() {}

    const char* getPluginName() const { return DCN_PLUGIN_NAME; }

    const char* getPluginVersion() const { return DCN_PLUGIN_VERSION; }

    nvinfer1::PluginFieldCollection* getFieldNames() {
        return &_mFC;
    }
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc);

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) {
        return new DCNDynamicPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

    const char* getPluginNamespace() const { return mNamespace.c_str(); }

};

/// register plugin
REGISTER_TENSORRT_PLUGIN(DCNDynamicPluginCreator);
// end of this file
#endif 
