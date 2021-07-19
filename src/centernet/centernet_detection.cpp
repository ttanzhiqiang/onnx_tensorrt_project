#include "assert.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <common.h>
#include "Trt.h"
#include "ctdetLayer.h"
#include "class_timer.hpp"
struct CenterNetResult
{
	int		 id = -1;
	float	 prob = 0.f;
	cv::Rect rect;
};
typedef std::vector<CenterNetResult> BatchResult;
class CenterNetDectector
{
public:
	std::shared_ptr<Trt> onnx_net;
	uint32_t m_InputH;
	uint32_t m_InputW;
	uint32_t m_InputC;
	uint32_t m_InputSize;
	uint32_t m_BatchSize = 1;
	float conf_thresh = 0.4;
	float m_NMSThresh = 0.2;
	int m_Classes;
	int m_kernelSize = 3;
	std::vector<std::string> m_ClassNames;
	std::vector<TensorInfo> m_OutputTensors;
	cudaStream_t mCudaStream;
	Config _config;
public:
	CenterNetDectector::CenterNetDectector()
	{

	}
	CenterNetDectector::~CenterNetDectector()
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

	std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
	{
		auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float
		{
			if (x1min > x2min)
			{
				std::swap(x1min, x2min);
				std::swap(x1max, x2max);
			}
			return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
		};
		auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float
		{
			float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
			float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
			float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
			float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
			float overlap2D = overlapX * overlapY;
			float u = area1 + area2 - overlap2D;
			return u == 0 ? 0 : overlap2D / u;
		};

		std::stable_sort(binfo.begin(), binfo.end(),
			[](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
		std::vector<BBoxInfo> out;
		for (auto& i : binfo)
		{
			bool keep = true;
			for (auto& j : out)
			{
				if (keep)
				{
					float overlap = computeIoU(i.box, j.box);
					keep = overlap <= nmsThresh;
				}
				else
					break;
			}
			if (keep) out.push_back(i);
		}
		return out;
	}

	std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh,
		std::vector<BBoxInfo>& binfo,
		const uint32_t numClasses,
		const std::string& model_type)
	{
		std::vector<BBoxInfo> result;
		std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
		for (auto& box : binfo)
		{
			splitBoxes.at(box.label).push_back(box);
		}

		for (auto& boxes : splitBoxes)
		{
			boxes = nonMaximumSuppression(nmsThresh, boxes);
			result.insert(result.end(), boxes.begin(), boxes.end());
		}

		return result;
	}

	float getNMSThresh() const { return m_NMSThresh; }

	uint32_t getNumClasses() const { return static_cast<uint32_t>(m_ClassNames.size()); }

	void UpdateOutputTensor()
	{
		m_InputW = onnx_net->mBindingDims[0].d[2];
		m_InputH = onnx_net->mBindingDims[0].d[3];
		m_Classes = onnx_net->mBindingDims[1].d[1];
		for (int m_centernet_ind = 1; m_centernet_ind < onnx_net->mBindingName.size(); m_centernet_ind++)
		{
			TensorInfo outputTensor;
			outputTensor.volume = onnx_net->mBindingSize[m_centernet_ind] / sizeof(float);
			outputTensor.blobName = onnx_net->mBindingName[m_centernet_ind];
			m_OutputTensors.push_back(outputTensor);
		}
	}

	void allocateBuffers()
	{
		for (auto& tensor : m_OutputTensors)
		{
			tensor.bindingIndex = onnx_net->mEngine->getBindingIndex(tensor.blobName.c_str());
			assert((tensor.bindingIndex != -1) && "Invalid output binding index");
			tensor.hostBuffer.resize(tensor.volume * m_BatchSize * sizeof(float));
		}
	}

	void isInt8(std::string calibration_image_list_file, int width, int height)
	{
		size_t npos = _config.onnxModelpath.find_first_of('.');
		std::string calib_table_name = _config.onnxModelpath.substr(0, npos) + ".table";
		if (!fileExists(calib_table_name))
		{
			onnx_net->SetInt8Calibrator("Int8MinMaxCalibrator", width, height, calibration_image_list_file.c_str(), calib_table_name.c_str());
		}
	}

	void init(Config config)
	{
		_config = config;
		onnx_net = std::make_shared<Trt>();
		onnx_net->SetMaxBatchSize(config.maxBatchSize);
		conf_thresh = config.conf_thresh;
		m_NMSThresh = config.m_NMSThresh;
		if (config.mode == 2)
		{
			isInt8(config.calibration_image_list_file, config.calibration_width, config.calibration_height);
		}
		m_BatchSize = config.maxBatchSize;
		onnx_net->CreateEngine(config.onnxModelpath, config.engineFile, config.customOutput, config.maxBatchSize, config.mode);
		//更新m_OutputTensors
		UpdateOutputTensor();
		allocateBuffers();
		cudaStreamCreate(&mCudaStream);
	}

	void doInference(std::vector<float> input, const uint32_t batchSize)
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

	std::vector<std::vector<BBoxInfo>> reprocessing()
	{
		Timer timer;
		std::vector<std::vector<BBoxInfo>> m_batch_box;
		m_batch_box.reserve(m_BatchSize);
		for (uint32_t i_BatchSize = 0; i_BatchSize < m_BatchSize; i_BatchSize++)
		{
			void* cudaOutputBuffer;
			int outputBufferSize = onnx_net->mBindingSize[1] * 6 / m_BatchSize;
			std::unique_ptr<float[]> outputData(new float[outputBufferSize]);
			cudaMalloc(&cudaOutputBuffer, outputBufferSize);
			CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
			const float* m_hm_hostBuffer = static_cast<const float*>(onnx_net->mBinding[1]) + i_BatchSize * m_OutputTensors[0].volume / m_BatchSize;
			const float* m_reg_hostBuffer = static_cast<const float*>(onnx_net->mBinding[2]) + i_BatchSize * m_OutputTensors[1].volume / m_BatchSize;
			const float* m_wh_hostBuffer = static_cast<const float*>(onnx_net->mBinding[3]) + i_BatchSize * m_OutputTensors[2].volume / m_BatchSize;
			CTdetforward_gpu(m_hm_hostBuffer, m_reg_hostBuffer, m_wh_hostBuffer, static_cast<float*>(cudaOutputBuffer),
				m_InputW / 4, m_InputH / 4, m_Classes, m_kernelSize, conf_thresh);
			//CUDA_CHECK(cudaMemcpyAsync(outputData.get(), cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mCudaStream));
			cudaMemcpyAsync(outputData.get(), cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mCudaStream);
			std::vector<BBoxInfo> result;
			int num_det = static_cast<int>(outputData[0]);
			result.resize(num_det);
			memcpy(result.data(), &outputData[1], num_det * sizeof(BBoxInfo));
			m_batch_box.push_back(result);
			cudaFree(cudaOutputBuffer);
		}
		return m_batch_box;
	}

	void postProcess(std::vector<BBoxInfo>& result, const cv::Mat& img)
	{
		using namespace cv;
		int input_w = m_InputW;
		int input_h = m_InputH;
		float scale_w = float(input_w) / img.cols;
		float scale_h = float(input_h) / img.rows;
		float dx = (input_w - scale_w * img.cols) / 2;
		float dy = (input_h - scale_h * img.rows) / 2;
		for (auto& item : result)
		{
			float x1 = (item.box.x1 - dx) / scale_w;
			float y1 = (item.box.y1 - dy) / scale_h;
			float x2 = (item.box.x2 - dx) / scale_w;
			float y2 = (item.box.y2 - dy) / scale_h;
			x1 = (x1 > 0) ? x1 : 0;
			y1 = (y1 > 0) ? y1 : 0;
			x2 = (x2 < img.cols) ? x2 : img.cols - 1;
			y2 = (y2 < img.rows) ? y2 : img.rows - 1;
			item.box.x1 = x1;
			item.box.y1 = y1;
			item.box.x2 = x2;
			item.box.y2 = y2;
		}
	}

	void detect(const std::vector<cv::Mat>& vec_image,
		std::vector<BatchResult>& vec_batch_result)
	{
		vec_batch_result.clear();
		vec_batch_result.reserve(vec_image.size());
		std::vector<float>data;
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
			data.insert(data.end(), ptr1, ptr1 + m_InputH * m_InputW);
			data.insert(data.end(), ptr2, ptr2 + m_InputH * m_InputW);
			data.insert(data.end(), ptr3, ptr3 + m_InputH * m_InputW);
		}
		Timer timer;
		timer.reset();
		doInference(data, vec_image.size());
		double t_doInference = timer.elapsed();
		std::cout << "doInference:"  << t_doInference << "ms" << std::endl;
		timer.reset();
		std::vector < std::vector<BBoxInfo>> m_batch_box = reprocessing();
		double reprocessing_t = timer.elapsed();
		std::cout << "reprocessing:" << reprocessing_t << "ms" << std::endl;
		for (uint32_t i = 0; i < vec_image.size(); ++i)
		{
			auto curImage = vec_image.at(i);
			auto binfo = m_batch_box.at(i);
			postProcess(binfo, curImage);
			auto remaining = nmsAllClasses(getNMSThresh(),
				binfo,
				m_Classes,
				"");
			if (remaining.empty())
			{
				continue;
			}
			std::vector<CenterNetResult> vec_result(0);
			vec_result.reserve(remaining.size());
			for (const auto& b : remaining)
			{
				CenterNetResult res;
				res.id = b.label;
				res.prob = b.prob;
				const int x = b.box.x1;
				const int y = b.box.y1;
				const int w = b.box.x2 - b.box.x1;
				const int h = b.box.y2 - b.box.y1;
				res.rect = cv::Rect(x, y, w, h);
				vec_result.push_back(res);
			}
			//vec_batch_result[i] = vec_result;
			vec_batch_result.push_back(vec_result);
		}
	}
};

int main_CenterNetDectector()
{
	CenterNetDectector m_CenterNetDectector;
	Config m_config;
	m_config.onnxModelpath = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_centernet\\ctdet_coco_dla_2x.onnx";
	m_config.engineFile = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_centernet\\ctdet_coco_dla_2x_fp32_batch_1.engine";
	m_config.calibration_image_list_file = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_centernet\\image\\";
	m_config.maxBatchSize = 1;
	m_config.mode = 1;
	m_config.calibration_width = 512;
	m_config.calibration_height = 512;
	m_config.conf_thresh = 0.5;
	m_config.m_NMSThresh = 0.2;
	m_CenterNetDectector.init(m_config);
	std::vector<BatchResult> batch_res;
	std::vector<cv::Mat> batch_img;
	std::string filename = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_centernet\\image\\17790319373_bd19b24cfc_k.jpg";
	cv::Mat image = cv::imread(filename);
	//std::string filename_1 = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\pytorch_onnx_tensorrt_centernet\\image\\dog.jpg";
	//cv::Mat image_1 = cv::imread(filename_1);
	batch_img.push_back(image);
	//batch_img.push_back(image_1);
	//float all_time = 0.0;
	float all_time = 0.0;
	time_t start = time(0);
	Timer timer;
	int m = 200;
	for (int i = 0; i < m; i++)
	{
		//timer.reset();
		clock_t start, end;
		timer.reset();
		m_CenterNetDectector.detect(batch_img, batch_res);
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
	
	//disp
	for (int i = 0; i < batch_img.size(); ++i)
	{
		for (const auto& r : batch_res[i])
		{
			std::cout << "batch " << i << " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
			cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
			cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		cv::imshow("image" + std::to_string(i), batch_img[i]);
	}
	cv::waitKey(10);
	return 0;
}