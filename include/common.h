#pragma once
#ifndef ENTROPY_COMMON
#define ENTROPY_COMMON
#include <experimental/filesystem>
#include <map>
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

#define ASSERT(assertion)                                                                                              \
{                                                                                                                  \
    if (!(assertion))                                                                                              \
    {                                                                                                              \
        std::cerr << "#assertion fail " << __FILE__ << " line " << __LINE__ << std::endl;                                     \
        abort();                                                                                                   \
    }                                                                                                              \
}

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::gLogError << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line "         \
                              << __LINE__ << std::endl;                                                                \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)


namespace Tn
{
    template<typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

typedef struct
{
    int r;              //red channel: [0,255]
    int g;             // green channel:[0, 255]	
    int b;                // blue channel:[0, 255]
}ColorPoint, * PColorPoint;
typedef std::map<std::string, ColorPoint> LabelNameColorMap;

#include <iostream>
#include <vector>
struct Config
{
    std::string cfgFile = "configs/yolov3.cfg";

    std::string onnxModelpath = "configs/yolov3.onnx";

    std::string engineFile = "configs/yolov3.engine";

    std::string calibration_image_list_file = "configs/images/";

    std::vector<std::string> customOutput;

    int calibration_width = 0;

    int calibration_height = 0;
    
    int maxBatchSize = 1;

    int mode; //0£¬1£¬2

    float conf_thresh = 0.6;
    //std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
    LabelNameColorMap ncp;
};

struct TensorInfo
{
    std::string blobName;
    uint32_t stride{ 0 };
    uint32_t stride_h{ 0 };
    uint32_t stride_w{ 0 };
    uint32_t gridSize{ 0 };
    uint32_t grid_h{ 0 };
    uint32_t grid_w{ 0 };
    uint32_t numClasses{ 0 };
    uint32_t numBBoxes{ 0 };
    uint64_t volume{ 0 };
    std::vector<uint32_t> masks;
    std::vector<float> anchors;
    int bindingIndex{ -1 };
    std::vector<float> hostBuffer;
    //float* hostBuffer;
};

struct BBox
{
    float x1, y1, x2, y2;
};

struct BBoxInfo
{
    BBox box;
    int label;
    int classId; // For coco benchmarking
    float prob;
};

#ifndef BLOCK
#define BLOCK 512
#endif

bool fileExists(const std::string fileName);
#endif // ENTROPY_COMMON