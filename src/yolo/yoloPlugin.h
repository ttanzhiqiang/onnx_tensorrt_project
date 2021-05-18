#ifndef TRT_Yolo_PLUGIN_H
#define TRT_Yolo_PLUGIN_H
#include <string>
#include <vector>
#include <iostream>
#include "NvInfer.h"

namespace nvinfer1
{
    class Yolo : public IPluginV2IOExt
    {
    public:
        explicit Yolo();
        Yolo(int numclass, int gride_w,int gride_h, int numanchors);

        Yolo(const void* buffer, size_t length);

        ~Yolo();

        int getNbOutputs() const override;

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        int initialize() override;

        void terminate() override;

        size_t getWorkspaceSize(int maxBatchSize) const override;

        int enqueue(
            int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        size_t getSerializationSize() const override;

        void serialize(void* buffer) const override;

        //bool supportsFormat(DataType type, PluginFormat format) const override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        IPluginV2IOExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

        //void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        //    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        //    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

        void detachFromContext() override;

    private:
        const char* mPluginNamespace;

        int numclass_;
        int numanchors_;
        int _n_grid_h;
        int _n_grid_w;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const PluginFieldCollection* getFieldNames() override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;


        virtual void setPluginNamespace(const char* pluginNamespace) override {
            mNamespace = pluginNamespace;
        }

        virtual const char* getPluginNamespace() const override {
            return mNamespace.c_str();
        }

    private:
        static PluginFieldCollection mFC;
        int numclass_;
        int numanchors_;
        int _n_grid_h;
        int _n_grid_w;
        static std::vector<PluginField> mPluginAttributes;
        std::string mNamespace;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
} // namespace nvinfer1

#endif // TRT_Yolo_PLUGIN_H
