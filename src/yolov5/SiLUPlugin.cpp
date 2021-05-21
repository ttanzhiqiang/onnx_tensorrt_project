
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

#include "SiLUPlugin.h"
#include "NvInfer.h"
#include "common.h"
using namespace nvinfer1;
using nvinfer1::plugin::SiLUPluginCreator;
using nvinfer1::plugin::SILU;

static const char* SILU_PLUGIN_VERSION{"1"};
static const char* SILU_PLUGIN_NAME{"SiLU"};
PluginFieldCollection SiLUPluginCreator::mFC{};
std::vector<PluginField> SiLUPluginCreator::mPluginAttributes;

// LeakyReLU {{{
SILU::SILU()
{
}

SILU::SILU(const void* buffer, size_t length)
{
    // const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    // mBatchDim = read<int>(d);
    // ASSERT(d == a + length);
    assert(length==sizeof(input_size_));
    input_size_ = *reinterpret_cast<const int*>(buffer);

}

int SILU::getNbOutputs() const
{
    return 1;
}

Dims SILU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
    return inputs[0];
}

//__device__ float Logist_kernel_cpu(float data) { return 1.0f / (1.0f + expf(-data)); };
//
//
//pluginStatus_t SILU::SiLUInference_cpu(const int n, const float* input, float* output)
//{
//    printf("SiLUInference_cpu start\n");
//    for (int i =0; i < n; i += 1)
//    {
//        printf("SiLUInference_cpu id=%d\n",i);
//        output[i] = input[i] * Logist_kernel_cpu(input[i]);
//    }
//    return STATUS_SUCCESS;
//}


int SILU::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = SiLUInference(stream, batchSize*input_size_, inputData, outputData);
    // pluginStatus_t status = SiLUInference_cpu(batchSize*input_size_, (const float*) inputData, (float*) outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t SILU::getSerializationSize() const
{
    // mNegSlope, mBatchDim
    // return sizeof(float) + sizeof(int);
    // return sizeof(int);
    return sizeof(input_size_);
}

// Set plugin namespace
void SILU::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* SILU::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType SILU::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool SILU::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool SILU::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void SILU::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
{
    // ASSERT(mBatchDim == 1);
    // for (int i = 0; i <in->dims.nbDims; ++i)
    // {
    //     mBatchDim *= in->dims.d[i];
    // }
}

void SILU::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void SILU::detachFromContext() {}

void SILU::serialize(void* buffer) const
{
    // char *d = reinterpret_cast<char*>(buffer), *a = d;
    // write(d, mBatchDim);
    // ASSERT(d == a + getSerializationSize());
    *reinterpret_cast<int*>(buffer)=input_size_;
}

// void SILU::configureWithFormat(
//     const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
// {
//     ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
//     ASSERT(mBatchDim == 1);
//     ASSERT(nbOutputs == 1);
//     for (int i = 0; i < inputDims[0].nbDims; ++i)
//     {
//         mBatchDim *= inputDims[0].d[i];
//     }
// }

// bool SILU::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// }

bool SILU::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
//    ASSERT(mBatchDim == 1);
//    for (int i = 0; i <inOut->dims.nbDims; ++i)
//    {
//        mBatchDim *= inOut->dims.d[i];
//    }
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

int SILU::initialize()
{
    return 0;
}

void SILU::terminate() {}

size_t SILU::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

const char* SILU::getPluginType() const
{
    return SILU_PLUGIN_NAME;
}

const char* SILU::getPluginVersion() const
{
    return SILU_PLUGIN_VERSION;
}

void SILU::destroy()
{
    delete this;
}

IPluginV2IOExt* SILU::clone() const
{
    SILU* plugin = new SILU();
    plugin->input_size_ = input_size_;
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

SiLUPluginCreator::SiLUPluginCreator()
{
    // mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SiLUPluginCreator::getPluginName() const
{
    return SILU_PLUGIN_NAME;
}

const char* SiLUPluginCreator::getPluginVersion() const
{
    return SILU_PLUGIN_VERSION;
}

const PluginFieldCollection* SiLUPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2IOExt* SiLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    // const PluginField* fields = fc->fields;
    // ASSERT(fc->nbFields == 1);
    // ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    // negSlope = *(static_cast<const float*>(fields[0].data));

    // return new SILU();
    SILU* obj = new SILU();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2IOExt* SiLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    // return new SILU(serialData, serialLength);
    SILU* obj = new SILU(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
// LeakReLU }}}
