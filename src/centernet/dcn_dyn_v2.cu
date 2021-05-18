#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "dcn_dyn_v2.hpp"
#include "dcn_v2_im2col_cuda.h"
#include <stdio.h>
#include <vector>
#include "NvInfer.h"

using namespace nvinfer1;
#define MAX_BATCH_SIZE 2
//#define USE_D_ONES

#define CHECK_CUDA(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
    exit(0); \
} \
}

#define CHECK_LAST_ERR(func) { \
    cudaError_t e = cudaGetLastError();\
    if (e != cudaSuccess) {\
        printf("cuda failure of %s: %s:%d: '%s'\n", func, __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(-1); \
    } \
}

/// Static class fields initialization
nvinfer1::PluginFieldCollection DCNDynamicPluginCreator::_mFC{};
std::vector<nvinfer1::PluginField> DCNDynamicPluginCreator::_mPluginAttributes;

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

cublasHandle_t blas_handle() {
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    const int n = 0;
    //cudaError_t status = cudaGetDevice(&n);
    if(!init[n]) {
        cublasStatus_t st = cublasCreate(&handle[n]);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("blas_handle create failed! %s:%d, code:%s\n", __FILE__, __LINE__, cublasGetErrorString(st));
        }
        init[n] = 1;
    }
    return handle[n];
}


// bias: (k,), output: (batch_num, k, n)
__global__ void ones_mul_bias(const float* bias, float* output, 
        const int n, const int k, const int batch_num) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_num * n * k) return;

    const size_t col_id = (tid % (n * k)) / n;

    output[tid] = bias[col_id];
} 

template <typename T>
void createBatchBuffers(T* buff[], T* data, const size_t len_per_batch, const int batch_num) {
    for(int i = 0; i < batch_num; ++i) {
        buff[i] = data + len_per_batch * i;
    }
}


__inline__ size_t divUp(size_t num, int threads) {
    return (num + threads - 1) / threads;
}

int DCNDynamicPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) {

    /// input' shape is CHW
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    const int in_batchs = input_dims.d[0];
    const int in_channels = input_dims.d[1];
    const int input_height = input_dims.d[2];
    const int input_width = input_dims.d[3];

    const int out_batchs = output_dims.d[0];
    const int out_channels = output_dims.d[1];
    const int output_height = output_dims.d[2];
    const int output_width = output_dims.d[3];

    ///
    const float* input = static_cast<const float*>(inputs[0]);
    const float* offset = static_cast<const float*>(inputs[1]);
    const float* mask = static_cast<const float*>(inputs[2]);
    const float* weight = static_cast<const float*>(inputs[3]);
    const float* bias = static_cast<const float*>(inputs[4]);
    float * output = static_cast<float *>(outputs[0]);
    cublasHandle_t handle = blas_handle();
    cublasSetStream(handle, stream);

    float alpha, beta;
    size_t m, n, k;
    m = out_channels;
    n = output_height * output_width;
    k = 1;
    alpha = 1.0;
    beta = 0.0;
    cublasStatus_t st;
    //assert(batchSize==1);
    int batchSize = 1;
    assert(batchSize <= MAX_BATCH_SIZE);

#ifdef USE_D_ONES
    st = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,&alpha,
            _d_ones, k, bias, k,&beta, output, n);
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm error occurred! %s : %d, error_code:%s, n:%d, m:%d, k:%d\n", __FILE__, __LINE__,
                cublasGetErrorString(st), n, m, k);
        exit(-1);
    }
#else
    size_t num_blocks = divUp(n * m * batchSize, 512);
    ones_mul_bias<<<num_blocks, 512, 0, stream>>>(bias, output, n, m, batchSize);
#endif
    // im2col (offset and mask)
    /// offset: (batch_num, o_c, in_h, in_w)
    /// mask:(batch_num, o_m, in_h, in_w)
    modulated_deformable_im2col_cuda(stream,input,offset,mask, \
            batchSize, in_channels, input_height, input_width,
            output_height, output_width, _kernel_size, _kernel_size,
            _padding, _padding, _stride, _stride, _dilation, _dilation,
            _deformable_groups, _d_columns);


    m = out_channels;
    n = output_height * output_width;
    k = in_channels * _kernel_size * _kernel_size;
    alpha = 1.0;
    beta = 1.0;
    // im2col conv
    /// _d_columns: batch* (k=in_c*ker*ker) * (n=o_h*o_w)
    /// weight:  m(o_c) * k
    /// output: batch*m*n
    /// output = weight x _d_columns

    // C^T = (AB)^T = B^T A^T
    // B_COL, A_ROW, B_ROW, d_B, B_COL, d_A, A_COL, d_C, B_COL
    //if (batchSize == 1) {
    //    st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,&alpha, _d_columns, n, weight, k,&beta, output, n);
    //} else
    {
        float* col_buff[MAX_BATCH_SIZE];
        const float* w_buff[MAX_BATCH_SIZE];
        float* o_buff[MAX_BATCH_SIZE];
        createBatchBuffers<float>(col_buff, _d_columns, n * k, batchSize);
        createBatchBuffers<float>(o_buff, output, n*m, batchSize);
        for(int i = 0; i < batchSize; ++i) w_buff[i] = weight;

        float** d_col_buff, **d_w_buff, **d_o_buff;
        cudaMalloc(&d_col_buff, sizeof(float*) * batchSize);
        cudaMalloc(&d_w_buff, sizeof(float*) * batchSize);
        cudaMalloc(&d_o_buff, sizeof(float*) * batchSize);
        cudaMemcpy(d_col_buff, col_buff, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w_buff, w_buff, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_o_buff, o_buff, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);

        st = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, &alpha, d_col_buff, n,
            d_w_buff, k, &beta, d_o_buff, n, batchSize);

    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm error occurred! %s : %d, error_code:%s, n:%d, m:%d, k:%d\n", __FILE__, __LINE__,
                cublasGetErrorString(st), n, m, k);
        exit(-1);
    }


    return 0;
}

int DCNDynamicPlugin::initialize()  {
    //std::cout << "**** initialize called! **** id:"  << this<<  std::endl;
    if(_initialized) return 0;
    auto _out_dims = this->getOutputDimensions(0, &this->getInputDims(0), 5);
    size_t ones_size = _out_dims.d[1]*_out_dims.d[2];
#ifdef USE_D_ONES    
    CHECK_CUDA(cudaMalloc((void**)&_d_ones, ones_size*sizeof(float) ));      
    float* cpu_ones = new float[ones_size];
    for(size_t i = 0; i < ones_size; ++i) cpu_ones[i] = 1.0;
    CHECK_CUDA(cudaMemcpy(_d_ones, cpu_ones, ones_size*sizeof(float), cudaMemcpyHostToDevice));
#endif     
    size_t column_size = _inputDims.d[0] * _kernel_size * _kernel_size * ones_size * MAX_BATCH_SIZE;
    CHECK_CUDA(cudaMalloc((void**)&_d_columns, sizeof(float) * column_size));

#ifdef USE_D_ONES
    delete [] cpu_ones;
#endif 
    _initialized = true;

    return 0;
}

void DCNDynamicPlugin::terminate()  {
    if(!_initialized) return;
#ifdef USE_D_ONES    
    CHECK_CUDA(cudaFree(_d_ones));
#endif     
    CHECK_CUDA(cudaFree(_d_columns));

    _initialized = false;
}


/*
IPluginV2DynamicExt getOutputDimensions
*/
nvinfer1::DimsExprs DCNDynamicPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
    assert(outputIndex == 0);
    assert(nbInputs == 5);
    nvinfer1::DimsExprs output(inputs[0]);
    auto input_h = output.d[2]->getConstantValue();
    auto input_w = output.d[3]->getConstantValue();
    auto output_h = (input_h + 2 * _padding - (_dilation * (_kernel_size - 1) + 1)) / _stride + 1;
    auto output_w = (input_w + 2 * _padding - (_dilation * (_kernel_size - 1) + 1)) / _stride + 1;
    output.d[1] = exprBuilder.constant(_outputDims.d[0]);
    output.d[2] = exprBuilder.constant(output_h);
    output.d[3] = exprBuilder.constant(output_w);
    return output;
}

/*
IPluginV2DynamicExt supportsFormatCombination
*/
bool DCNDynamicPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    assert(nbInputs == 5);
    assert(nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == nvinfer1::TensorFormat::kNCHW);
}

/*
IPluginV2DynamicExt configurePlugin
*/
void DCNDynamicPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
    assert(nbInputs == 5);
    assert(nbOutputs == 1);
    auto& input_desc = in[0].desc;
    auto input_dims = input_desc.dims;
    _outputDims.d[1] = (input_dims.d[2] + 2 * _padding - (_dilation * (_kernel_size - 1) + 1)) / _stride + 1;
    _outputDims.d[2] = (input_dims.d[3] + 2 * _padding - (_dilation * (_kernel_size - 1) + 1)) / _stride + 1;
}

DCNDynamicPluginCreator::DCNDynamicPluginCreator()
{
    //_mPluginAttributes.emplace_back(PluginField("in_channel", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("in_height", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("in_width", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("out_channel", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("out_height", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("out_width", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("deformable_group", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    //_mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kFLOAT32, 1));
    //_mFC.nbFields = _mPluginAttributes.size();
    //_mFC.fields = _mPluginAttributes.data();
}

IPluginV2DynamicExt* DCNDynamicPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::vector<float> weight;
    std::vector<float> bias;
    int in_width, in_height, out_width, out_height;
    int in_channel, out_channel, kernel, deformable_group, padding, stride, dilation;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "in_channel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            in_channel = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "out_channel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            out_channel = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "in_width"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            in_width = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "in_height"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            in_height = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "out_width"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            out_width = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "out_height"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            out_height = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "deformable_group"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            deformable_group = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "dilation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            dilation = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            stride = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            padding = *(static_cast<const int*>(fields[i].data));
        }
    }
    DCNDynamicPlugin* obj = new DCNDynamicPlugin(in_channel, out_channel,
        in_width, in_height, out_width, out_height,
        kernel, deformable_group,
        dilation, padding, stride);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}