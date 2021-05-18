#ifndef TRT_Yolo_PLUGIN_H
#define TRT_Yolo_PLUGIN_H
#include <string>
#include <vector>
#include <iostream>
#include "NvInfer.h"

namespace nvinfer1
{
    class YoloDynamic : public IPluginV2DynamicExt
    {
    public:
        explicit YoloDynamic();
        YoloDynamic(int numclass, int gride_w,int gride_h, int numanchors);

        YoloDynamic(const void* buffer, size_t length);

        ~YoloDynamic();

        int getNbOutputs() const override;

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        int initialize() override;

        void terminate() override;

        size_t getWorkspaceSize(int maxBatchSize) const override;

        int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
            void* const* outputs, void* workspace, cudaStream_t stream) override;

        size_t getSerializationSize() const override;

        void serialize(void* buffer) const override;

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        IPluginV2DynamicExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

        void detachFromContext() override;

        //IPluginV2DynamicExt
        nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
            const nvinfer1::DimsExprs* inputs,
            int nbInputs,
            nvinfer1::IExprBuilder& exprBuilder) override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
            const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override {};

        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
            return 0;
        }
    private:
        const char* mPluginNamespace;

        int numclass_;
        int numanchors_;
        int _n_grid_h;
        int _n_grid_w;
    };

    class YoloDynamicPluginCreator : public IPluginCreator
    {
    public:
        YoloDynamicPluginCreator();

        ~YoloDynamicPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const PluginFieldCollection* getFieldNames() override;

        IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

        IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;


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
    REGISTER_TENSORRT_PLUGIN(YoloDynamicPluginCreator);
} // namespace nvinfer1

#endif // TRT_Yolo_PLUGIN_H
