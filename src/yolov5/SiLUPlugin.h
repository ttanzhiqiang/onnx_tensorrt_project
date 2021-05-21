/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_SILU_PLUGIN_H
#define TRT_SILU_PLUGIN_H
#include "NvInfer.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

pluginStatus_t SiLUInference(cudaStream_t stream, int n, const void* input, void* output);

namespace nvinfer1
{
namespace plugin
{

class SILU : public IPluginV2IOExt
{
public:
    SILU();

    SILU(const void* buffer, size_t length);

    ~SILU() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;

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

    void detachFromContext() override;

    int input_size_;

private:
    const char* mPluginNamespace;
    // int mBatchDim;
    //pluginStatus_t SiLUInference_cpu(const int n, const float* input, float* output);

};

class SiLUPluginCreator : public IPluginCreator
{
public:
    SiLUPluginCreator();

    ~SiLUPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }


private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(SiLUPluginCreator);

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SILU_PLUGIN_H
