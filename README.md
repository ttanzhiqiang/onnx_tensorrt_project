# ONNX-TensorRT

# Yolov5(4.0)/Yolov5(5.0)/Yolov4/Yolov3/CenterNet/Classify/Unet Implementation



Yolov4/Yolov3

- ![](./model/result/yolo_result.png)
-  

![](./model/result/yolo_result0.png)





centernet

![](./model/result/centernet_result.png)



## INTRODUCTION

you have the trained model file from the darknet/libtorch/pytorch

![](./model/result/onnx_trt.jpg)

- [x] yolov5-4.0(5s/5m/5s/5x)
- [x] yolov5-5.0(5s/5m/5s/5x)
- [x] yolov4 , yolov4-tiny
- [x] yolov3 , yolov3-tiny
- [x] centernet
- [x] classify(mnist\alexnet\resnet18\resnet34\resnet50\shufflenet_v2\mobilenet_v2)

## Features

- [x] inequal net width and height

- [x] batch inference

  ------

  onnx-tensorrt batch inference : onnx re-export(batch:2)

- [x] support FP32,FP16,INT8

- [x] dynamic input size(tiny_tensorrt_dyn_onnx)

<details><summary><b>BENCHMARK</b></summary>

#### window x64 (detect time)


|   model   |  size   |  gpu   |  fp32   |  fp16   |  INT8   |
| :-------: | :-----: | :----: | :-----: | :-----: | :-----: |
|  yolov3   | 608x608 | 2080ti | 28.14ms | 19.79ms | 18.53ms |
|  yolov4   | 320x320 | 2080ti | 8.85ms  | 6.62ms  | 6.33ms  |
|  yolov4   | 416x416 | 2080ti | 12.19ms | 10.20ms | 9.35ms  |
|  yolov4   | 512x512 | 2080ti | 15.63ms | 12.66ms | 12.19ms |
|  yolov4   | 608x608 | 2080ti | 24.39ms | 17.54ms | 17.24ms |
|  yolov4   | 320x320 |  3070  | 9.70ms  | 7.30ms  | 6.37ms  |
|  yolov4   | 416x416 |  3070  | 14.08ms | 9.80ms  | 9.70ms  |
|  yolov4   | 512x512 |  3070  | 18.87ms | 13.51ms | 13.51ms |
|  yolov4   | 608x608 |  3070  | 28.57ms | 19.60ms | 18.52ms |
|  yolov4   | 320x320 |  1070  | 18.52ms |    \    | 12.82ms |
|  yolov4   | 416x416 |  1070  | 27.03ms |    \    | 20.83ms |
|  yolov4   | 512x512 |  1070  | 34.48ms |    \    | 27.03ms |
|  yolov4   | 608x608 |  1070  |  50ms   |    \    | 35.71ms |
|  yolov4   | 320x320 | 1660TI | 16.39ms | 11.90ms | 10.20ms |
|  yolov4   | 416x416 | 1660TI | 23.25ms | 17.24ms | 13.70ms |
|  yolov4   | 512x512 | 1660TI | 29.41ms | 24.39ms | 21.27ms |
|  yolov4   | 608x608 | 1660TI | 43.48ms | 34.48ms | 26.32ms |
| yolov5 5s | 608x608 | 2080ti | 24.47ms | 22.46ms |    /    |
| yolov5 5m | 608x608 | 2080ti | 30.61ms | 24.02ms |    /    |
| yolov5 5l | 608x608 | 2080ti | 32.58ms | 25.84ms |    /    |
| yolov5 5x | 608x608 | 2080ti | 40.69ms | 29.81ms |    /    |
| darknet53 | 224*224 | 2080ti | 3.53ms  | 1.84ms  | 1.71ms  |
| darknet53 | 224*224 |  3070  | 4.29ms  | 2.16ms  | 1.75ms  |



#### x64(inference / detect time)

|   model   |  size   |  gpu   | fp32(inference/detect) | fp16(inference/detect) | INT8(inference/detect) |
| :-------: | :-----: | :----: | :--------------------: | :--------------------: | :--------------------: |
| centernet | 512x512 | 2080ti |     17.8ms/39.7ms      |     15.7ms/36.49ms     |    14.37ms/36.34ms     |

</details>

## windows10

- dependency : spdlog，onnx，onnx-tensorrt，protobuf-3.11.4，TensorRT 7.2.2.3  , cuda 11.1 , cudnn 8.0  , opencv3.4, vs2019

- build:

    open MSVC _tiny_tensorrt_onnx.sln_ file 

    tiny_tensorrt_dyn_onnx:dynamic shape 

    tiny_tensorrt_onnx: normal

- build onnx-tensorrt

    step1: https://github.com/onnx/onnx-tensorrt.git

    step2: https://drive.google.com/drive/folders/1DndiqyCZ796p3-xXI3O4AMCIGcUWQ1q2?usp=sharing or https://share.weiyun.com/CJCwngAM

    step3: **builtin_op_importers.cpp** replace onnx-tensorrt\builtin_op_importers.cpp

    step4: tortoiseGit->apply patch serial and choose **0001-Compile-onnx-tensorrt-by-MSVC-on-Windows.patch**

    step5:build onnx.lib\onnx_proto.lib\nvonnxparser.dll\nvonnxparser_static.lib

## Model and 3rdparty

model : https://drive.google.com/drive/folders/1KzBjmCOG9ghcq9L6-iqfz6QwBQq6Hl4_?usp=sharing or https://share.weiyun.com/td9CRDhW

3rdparty:https://drive.google.com/drive/folders/1SddUgQ5kGlv6dDGPqnVWZxgCoBY85rM2?usp=sharing or https://share.weiyun.com/WEZ3TGtb

## API

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
	
	    int mode; //0，1，2
	
	    //std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
	};
	
	class YoloDectector
	{
	void init(Config config);
	void detect(const std::vector<cv::Mat>& vec_image,
		std::vector<BatchResult>& vec_batch_result);
	}

## REFERENCE

https://github.com/onnx/onnx-tensorrt.git

https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleDynamicReshape

https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps

https://github.com/enazoe/yolo-tensorrt.git

https://github.com/zerollzeng/tiny-tensorrt.git
## Contact

<img src="./model/result/weixin.jpg" style="zoom:50%;" />
