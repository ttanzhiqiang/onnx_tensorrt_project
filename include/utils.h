/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2019-11-12 11:53:56
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-06 17:17:13
 */
#ifndef UTILS_H
#define UTILS_H
#include <dirent.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "cuda_runtime.h"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#define UNUSED(unusedVariable) (void)(unusedVariable)
// suppress compiler warning: unused parameter

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        default: throw std::runtime_error("Invalid DataType.");
    }
}


#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(0);                                                                         \
        }                                                                                      \
    }
#endif

inline void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline void safeCudaFree(void* deviceMem) {
    CUDA_CHECK(cudaFree(deviceMem));
}

inline void error(const std::string& message, const int line, const std::string& function, const std::string& file) {
    std::cout << message << " at " << line << " in " << function << " in " << file << std::endl;
}
#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) COMPILE_TEMPLATE_BASIC_TYPES(className, class)
#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) COMPILE_TEMPLATE_BASIC_TYPES(className, struct)
#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
    template classType  className<char>; \
    template classType  className<signed char>; \
    template classType  className<short>; \
    template classType  className<int>; \
    template classType  className<long>; \
    template classType  className<long long>; \
    template classType  className<unsigned char>; \
    template classType  className<unsigned short>; \
    template classType  className<unsigned int>; \
    template classType  className<unsigned long>; \
    template classType  className<unsigned long long>; \
    template classType  className<float>; \
    template classType  className<double>; \
    template classType  className<long double>

// const auto CUDA_NUM_THREADS = 512u;
// inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired,
//     const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
// {
// return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
// }



struct YoloKernel;

static constexpr int LOCATIONS = 4;
struct alignas(float) Detection{
    //x y w h
    float bbox[LOCATIONS];
    //float objectness;
    int classId;
    float prob;
};

static inline cv::Mat preprocess_img(cv::Mat img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

#endif