#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "Trt.h"
#include "common.h"
#include "class_timer.hpp"
using BatchResult = std::vector<cv::Mat>;
struct  Detection {
	//std::vector<float> mask;
	//todo:
	std::vector<unsigned char> mask;
};

class UnetParser
{
public:
	Config _config;
	uint32_t m_InputH;
	uint32_t m_InputW;
	uint32_t m_InputC;
	uint32_t m_InputSize;
	uint32_t m_BatchSize = 1;
	cudaStream_t mCudaStream;
	std::shared_ptr<Trt> onnx_net;
	std::vector<std::string> m_ClassNames;
	std::vector<TensorInfo> m_OutputTensors;
	float CONF_THRESH = 0.6;
	LabelNameColorMap ncp;
public:

	UnetParser::UnetParser()
	{

	}

	UnetParser::~UnetParser()
	{
		//释放内存
		m_ClassNames.clear();
		for (int i = 0; i < m_OutputTensors.size(); i++)
		{
			m_OutputTensors[i].hostBuffer.clear();
		}
		m_OutputTensors.clear();
		cudaStreamDestroy(mCudaStream);
	}

	void UnetParser::UpdateOutputTensor()
	{
		m_InputW = onnx_net->mBindingDims[0].d[2];
		m_InputH = onnx_net->mBindingDims[0].d[3];
		for (int m_yolo_ind = 1; m_yolo_ind < onnx_net->mBindingName.size(); m_yolo_ind++)
		{
			TensorInfo outputTensor;
			outputTensor.volume = onnx_net->mBindingSize[m_yolo_ind] / sizeof(float);
			outputTensor.blobName = onnx_net->mBindingName[m_yolo_ind];
			m_OutputTensors.push_back(outputTensor);
		}
	}

	void UnetParser::allocateBuffers()
	{
		for (auto& tensor : m_OutputTensors)
		{
			tensor.bindingIndex = onnx_net->mEngine->getBindingIndex(tensor.blobName.c_str());
			assert((tensor.bindingIndex != -1) && "Invalid output binding index");
			tensor.hostBuffer.resize(tensor.volume * m_BatchSize /** sizeof(float)*/);
		}
	}

	void UnetParser::doInference(std::vector<float> input, const uint32_t batchSize)
	{
		//	Timer timer;
		assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
		onnx_net->CopyFromHostToDevice(input, 0, mCudaStream);
		onnx_net->ForwardAsync(mCudaStream);
		for (auto& tensor : m_OutputTensors)
		{
			onnx_net->CopyFromDeviceToHost(tensor.hostBuffer, tensor.bindingIndex, mCudaStream);
		}
		cudaStreamSynchronize(mCudaStream);
	}

	std::vector<float> UnetParser::postprocess(std::vector<float> buffer)
	{
		// Softmax function
		std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](float val) {return std::exp(val); });
		float sum = std::accumulate(buffer.begin(), buffer.end(), 0.0);
		if (sum > 0)
		{
			for (int i = 0; i < buffer.size(); i++)
			{
				buffer[i] /= sum;
			}
		}
		return buffer;
	}

	float UnetParser::sigmoid(float x)
	{
		return (1 / (1 + exp(-x)));
	}

	void UnetParser::process_cls_result(Detection& res, float* output, int output_size) {
		//支持多类语义分割（返回标签图）:存在锯齿
		int m_classes = output_size / (m_InputW * m_InputH);
		res.mask.resize(m_InputW * m_InputH * 1);   //返回标签图
		for (int i = 0; i < m_InputW * m_InputH; ++i)
		{
			float max_pixel_value = -9999999999;
			for (int c = 0; c < m_classes; ++c)
			{
				if (*(output + i + m_InputW * m_InputH * c) >= max_pixel_value)
				{
					max_pixel_value = *(output + i + m_InputW * m_InputH * c);
					res.mask[i] = c;
				}
			}
		}
	}

	void UnetParser::isInt8(std::string calibration_image_list_file, int width, int height)
	{
		size_t npos = _config.onnxModelpath.find_first_of('.');
		std::string calib_table_name = _config.onnxModelpath.substr(0, npos) + ".table";
		if (!fileExists(calib_table_name))
		{
			onnx_net->SetInt8Calibrator("Int8MinMaxCalibrator", width, height, calibration_image_list_file.c_str(), calib_table_name.c_str());
		}
	}

	void UnetParser::init(Config config)
	{
		_config = config;
		onnx_net = std::make_shared<Trt>();
		onnx_net->SetMaxBatchSize(config.maxBatchSize);
		if (config.mode == 2)
		{
			isInt8(config.calibration_image_list_file, config.calibration_width, config.calibration_height);
		}
		m_BatchSize = config.maxBatchSize;
		//todo:
		this->ncp = config.ncp;
		this->CONF_THRESH = config.conf_thresh;
		onnx_net->CreateEngine(config.onnxModelpath, config.engineFile, config.customOutput, config.maxBatchSize, config.mode);
		//更新m_OutputTensors
		UpdateOutputTensor();
		allocateBuffers();
		cudaStreamCreate(&mCudaStream);
	}

	void UnetParser::detect(const std::vector<cv::Mat>& vec_image, std::vector<BatchResult>& vec_batch_result)
	{
		vec_batch_result.clear();
		vec_batch_result.resize(vec_image.size());
		std::vector<float> data;
		for (const auto& img : vec_image)
		{
			cv::Mat resized, imgf;
			cv::resize(img, resized, cv::Size(m_InputH, m_InputW));
			resized.convertTo(imgf, CV_32FC3, 1 / 255.0);
			std::vector<cv::Mat>channles(3);
			cv::split(imgf, channles);
			float* ptr1 = (float*)(channles[0].data);
			float* ptr2 = (float*)(channles[1].data);
			float* ptr3 = (float*)(channles[2].data);

			//BGRBGRBGR->BBBGGGRRR
			data.insert(data.end(), ptr1, ptr1 + m_InputH * m_InputW);
			data.insert(data.end(), ptr2, ptr2 + m_InputH * m_InputW);
			data.insert(data.end(), ptr3, ptr3 + m_InputH * m_InputW);
		}
		doInference(data, vec_image.size());



		for (int i = 0; i < vec_image.size(); i++)
		{
			float max_conf = 0.0;
			int max_indice;
			Detection m_Detection;
			//
			for (auto& tensor : m_OutputTensors)
			{
				//通过设定阈值进行结果输出
				int m_classes = tensor.hostBuffer.size() / (m_InputW * m_InputH);
				m_Detection.mask.resize(m_InputW * m_InputH * 1);   //返回类标签图
				for (int i = 0; i < m_InputW * m_InputH; i++)
				{
					float max_pixel_value = -9999999999;
					m_Detection.mask[i] = 0;
					for (int c = 1; c < m_classes; ++c)
					{
						float pixel = sigmoid(*(tensor.hostBuffer.data() + i + m_InputW * m_InputH * c));
						if ((*(tensor.hostBuffer.data() + i + m_InputW * m_InputH * c) >= max_pixel_value) && pixel >= CONF_THRESH)
						{
							max_pixel_value = *(tensor.hostBuffer.data() + i + m_InputW * m_InputH * c);
							m_Detection.mask[i] = c;
						}
					}
				}
			}
			//float* mask = m_Detection.mask.data();
			unsigned char* mask = m_Detection.mask.data();
			cv::Mat mask_mat = cv::Mat(m_InputH, m_InputW, CV_8UC1);
			mask_mat.data = mask;
			cv::Mat mask_mat_rgb = cv::Mat(m_InputH, m_InputW, CV_8UC3);

			std::map<int, ColorPoint> int_map;
			int k = 0;
			for (auto a : ncp)
			{
				int_map[k] = a.second;
				++k;
			}

			for (int i = 0; i < m_InputH * m_InputW; ++i)
			{
				ColorPoint temp = int_map[(int)mask_mat.data[i]];
				mask_mat_rgb.data[3 * i] = temp.b;
				mask_mat_rgb.data[3 * i + 1] = temp.g;
				mask_mat_rgb.data[3 * i + 2] = temp.r;
			}
			vec_batch_result[i].push_back(mask_mat_rgb);
		}
	}
};

int main()
{

	UnetParser m_Unet;
	Config m_config;
	m_config.onnxModelpath = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_unet\\unet_three.onnx";
	m_config.engineFile = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_unet\\unet_three_int8_batch_1.engine";
	m_config.calibration_image_list_file = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_unet\\images\\";
	m_config.calibration_width = 512;
	m_config.calibration_height = 512;
	m_config.maxBatchSize = 1;
	m_config.mode = 2;
	m_config.conf_thresh = 0.8;
	m_config.ncp["background"] = ColorPoint{ 0,0,0 };
	//ncp["aeroplane"] = ColorPoint{ 128, 0, 0 };
	//ncp["bicycle"] = ColorPoint{ 0, 128, 0 };
	//ncp["bird"] = ColorPoint{ 128, 128, 0 };
	//ncp["boat"] = ColorPoint{ 0, 0, 128 };
	//ncp["bottle"] = ColorPoint{ 128, 0, 128 };
	//ncp["bus"] = ColorPoint{ 0, 128, 128 };
	//ncp["car"] = ColorPoint{ 128, 128, 128 };
	//ncp["cat"] = ColorPoint{ 64, 0, 0 };
	//ncp["chair"] = ColorPoint{ 192, 0, 0 };
	//ncp["cow"] = ColorPoint{ 64, 128, 0 };
	//ncp["diningtable"] = ColorPoint{ 192, 128, 0 };
	//ncp["dog"] = ColorPoint{ 64, 0, 128 };
	//ncp["horse"] = ColorPoint{ 192, 0, 128 };
	//ncp["motorbike"] = ColorPoint{ 64, 128, 128 };
	m_config.ncp["person"] = ColorPoint{ 192, 128, 128 };
	//ncp["pottedplant"] = ColorPoint{ 0, 64, 0 };
	//ncp["sheep"] = ColorPoint{ 128, 64, 0 };
	//ncp["sofa"] = ColorPoint{ 10, 192, 0 };
	m_config.ncp["train"] = ColorPoint{ 128, 192, 0 };
	//ncp["tvmonitor"] = ColorPoint{ 0, 64, 128 };

	m_Unet.init(m_config);
	std::vector<BatchResult> batch_res;
	std::vector<cv::Mat> batch_img;
	std::string filename = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\unet\\images\\000346.jpg";
	cv::Mat image = cv::imread(filename);
	batch_img.push_back(image);

	float all_time = 0.0;
	time_t start = time(0);
	Timer timer;
	int m = 100;
	for (int i = 0; i < m; i++)
	{
		//timer.reset();
		clock_t start, end;
		timer.reset();
		m_Unet.detect(batch_img, batch_res);
		double t = timer.elapsed();
		std::cout << i << ":" << t << "ms" << std::endl;
		if (i > 0)
		{
			all_time += t;
		}
	}
	std::cout << m << "次 time:" << all_time << " ms" << std::endl;
	std::cout << "1次 time:" << all_time / m << " ms" << std::endl;
	std::cout << "FPS::" << 1000 / (all_time / m) << std::endl;

	cv::Mat result;;
	cv::resize(batch_res[0][0], result, cv::Size(image.cols, image.rows));
	cv::imwrite("D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\unet\\output\\unet.png", result);
	return 0;
}