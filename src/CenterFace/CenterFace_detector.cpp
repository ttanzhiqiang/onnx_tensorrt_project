#include "assert.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <common.h>
#include "Trt.h"
#include "class_timer.hpp"
struct Result
{
	float	 prob = 0.f;
	cv::Rect rect;
	std::vector<cv::Point2f> keypoints;
};
typedef std::vector<Result> BatchResult;
class CenterFaceDectector
{
public:
	std::shared_ptr<Trt> onnx_net;
	uint32_t m_InputH;
	uint32_t m_InputW;
	uint32_t m_InputC;
	uint32_t m_InputSize;
	int m_Classes;
	uint32_t m_BatchSize = 1;
	float m_NMSThresh = 0.2;
	float obj_threshold = 0.2;
	int _n_yolo_ind = 0;
	std::vector<TensorInfo> m_OutputTensors;
	std::vector<std::map<std::string, std::string>> m_configBlocks;
	cudaStream_t mCudaStream;
	Config _config;
public:
	CenterFaceDectector::CenterFaceDectector()
	{

	}
	CenterFaceDectector::~CenterFaceDectector()
	{
		//释放内存
		for (int i = 0; i < m_OutputTensors.size(); i++)
		{
			m_OutputTensors[i].hostBuffer.clear();
		}
		m_OutputTensors.clear();
		m_configBlocks.clear();
		cudaStreamDestroy(mCudaStream);
	}

	struct FaceBox {
		float x;
		float y;
		float w;
		float h;
	};

	struct FaceRes {
		float confidence;
		FaceBox face_box;
		std::vector<cv::Point2f> keypoints;
	};

	float IOUCalculate(const FaceBox& det_a, const FaceBox& det_b) {
		cv::Point2f center_a(det_a.x, det_a.y);
		cv::Point2f center_b(det_b.x, det_b.y);
		cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
			std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
		cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
			std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
		float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
		float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
		float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
		float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
		float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
		float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
		if (inter_b < inter_t || inter_r < inter_l)
			return 0;
		float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
		float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
		if (union_area == 0)
			return 0;
		else
			return inter_area / union_area - distance_d / distance_c;
	}

	void NmsDetect(std::vector<FaceRes>& detections) {
		sort(detections.begin(), detections.end(), [=](const FaceRes& left, const FaceRes& right) {
			return left.confidence > right.confidence;
		});

		for (int i = 0; i < (int)detections.size(); i++)
			for (int j = i + 1; j < (int)detections.size(); j++)
			{
				float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
				if (iou > m_NMSThresh)
					detections[j].confidence = 0;
			}

		detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceRes& det)
		{ return det.confidence == 0; }), detections.end());
	}

	std::vector<std::vector<FaceRes>> postProcess(const std::vector<cv::Mat>& vec_Mat,
		float* output_1, float* output_2, float* output_3, float* output_4,
		const int& outSize_1, const int& outSize_2, const int& outSize_3, const int& outSize_4) {
		std::vector<std::vector<FaceRes>> vec_result;
		int index = 0;
		for (const cv::Mat& src_img : vec_Mat)
		{
			std::vector<FaceRes> result;
			int image_size = m_InputW / 4 * m_InputH / 4;
			float ratio = float(src_img.cols) / float(m_InputW) > float(src_img.rows) / float(m_InputH) ? float(src_img.cols) / float(m_InputW) : float(src_img.rows) / float(m_InputH);
			float* score = output_1 + index * outSize_1;
			float* scale0 = output_2 + index * outSize_2;
			float* scale1 = scale0 + image_size;
			float* offset0 = output_3 + index * outSize_3;
			float* offset1 = offset0 + image_size;
			float* landmark = output_4 + index * outSize_4;
			for (int i = 0; i < m_InputH / 4; i++) {
				for (int j = 0; j < m_InputW / 4; j++) {
					int current = i * m_InputW / 4 + j;
					if (score[current] > obj_threshold) {
						FaceRes headbox;
						headbox.confidence = score[current];
						headbox.face_box.h = std::exp(scale0[current]) * 4 * ratio;
						headbox.face_box.w = std::exp(scale1[current]) * 4 * ratio;
						headbox.face_box.x = ((float)j + offset1[current] + 0.5f) * 4 * ratio;
						headbox.face_box.y = ((float)i + offset0[current] + 0.5f) * 4 * ratio;
						for (int k = 0; k < 5; k++)
							headbox.keypoints.emplace_back(cv::Point2f(headbox.face_box.x - headbox.face_box.w / 2 + landmark[(2 * k + 1) * image_size + current] * headbox.face_box.w,
								headbox.face_box.y - headbox.face_box.h / 2 + landmark[(2 * k) * image_size + current] * headbox.face_box.h));
						result.push_back(headbox);
					}
				}
			}
			NmsDetect(result);
			vec_result.push_back(result);
			index++;
		}
		return vec_result;
	}

	void UpdateOutputTensor()
	{
		m_InputC = onnx_net->mBindingDims[0].d[1];
		m_InputW = onnx_net->mBindingDims[0].d[2];
		m_InputH = onnx_net->mBindingDims[0].d[3];
		m_Classes = onnx_net->mBindingDims[1].d[4] - 5;
		for (int m_yolo_ind = 1; m_yolo_ind < onnx_net->mBindingName.size(); m_yolo_ind++)
		{
			TensorInfo outputTensor;
			if (m_yolo_ind < 4)
			{
				outputTensor.masks = std::vector<uint32_t>{ uint32_t(3 * (m_yolo_ind - 1)),uint32_t(3 * (m_yolo_ind - 1) + 1),uint32_t(3 * (m_yolo_ind - 1) + 2) };
				outputTensor.numBBoxes = onnx_net->mBindingDims[m_yolo_ind].d[1];
				outputTensor.grid_w = onnx_net->mBindingDims[m_yolo_ind].d[2];
				outputTensor.grid_h = onnx_net->mBindingDims[m_yolo_ind].d[3];
				outputTensor.numClasses = onnx_net->mBindingDims[m_yolo_ind].d[4] - 5;
				outputTensor.stride_h = m_InputH / outputTensor.grid_h;
				outputTensor.stride_w = m_InputW / outputTensor.grid_w;
			}
			outputTensor.volume = onnx_net->mBindingSize[m_yolo_ind] / sizeof(float) / onnx_net->mBindingDims[m_yolo_ind].d[0];
			outputTensor.blobName = onnx_net->mBindingName[m_yolo_ind];
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
		if (config.mode == 2)
		{
			isInt8(config.calibration_image_list_file, config.calibration_width, config.calibration_height);
		}
		m_BatchSize = config.maxBatchSize;
		obj_threshold = config.conf_thresh;
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

	std::vector<float> prepareImage(std::vector<cv::Mat> vec_img) {
		std::vector<float> result(m_BatchSize * m_InputW * m_InputH * m_InputC);
		float* data = result.data();
		int index = 0;
		for (const cv::Mat& src_img : vec_img)
		{
			if (!src_img.data)
				continue;
			float ratio = float(m_InputW) / float(src_img.cols) < float(m_InputH) / float(src_img.rows) ? float(m_InputW) / float(src_img.cols) : float(m_InputH) / float(src_img.rows);
			cv::Mat flt_img = cv::Mat::zeros(cv::Size(m_InputW, m_InputH), CV_8UC3);
			cv::Mat rsz_img;
			cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
			rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
			flt_img.convertTo(flt_img, CV_32FC3);

			//HWC TO CHW
			int channelLength = m_InputW * m_InputH;
			std::vector<cv::Mat> split_img = {
					cv::Mat(m_InputW, m_InputH, CV_32FC1, data + channelLength * (index + 2)),
					cv::Mat(m_InputW, m_InputH, CV_32FC1, data + channelLength * (index + 1)),
					cv::Mat(m_InputW, m_InputH, CV_32FC1, data + channelLength * index)
			};
			index += 3;
			cv::split(flt_img, split_img);
		}
		return result;
	}

	void detect(const std::vector<cv::Mat>& vec_image,
		std::vector<BatchResult>& vec_batch_result)
	{
		vec_batch_result.clear();
		vec_batch_result.reserve(vec_image.size());
		std::vector<float> data = prepareImage(vec_image);
		Timer timer;
		timer.reset();
		doInference(data, vec_image.size());
		double t = timer.elapsed();
		std::cout << "doInference:" << t << "ms" << std::endl;
		auto faces = postProcess(vec_image, m_OutputTensors[0].hostBuffer.data(), m_OutputTensors[1].hostBuffer.data(),
			m_OutputTensors[2].hostBuffer.data(), m_OutputTensors[3].hostBuffer.data(),
			m_OutputTensors[0].hostBuffer.size(), m_OutputTensors[1].hostBuffer.size(),
			m_OutputTensors[2].hostBuffer.size(), m_OutputTensors[3].hostBuffer.size());
		for (uint32_t i = 0; i < vec_image.size(); ++i)
		{
			auto remaining = faces[i];
			if (remaining.empty())
			{
				continue;
			}
			std::vector<Result> vec_result(0);
			for (const auto& b : remaining)
			{
				Result res;
				res.prob = b.confidence;
				cv::Rect box(b.face_box.x - b.face_box.w / 2, b.face_box.y - b.face_box.h / 2, b.face_box.w, b.face_box.h);
				res.rect = box;
				res.keypoints = b.keypoints;
				vec_result.push_back(res);
			}
			vec_batch_result.push_back(vec_result);
		}
	}
};

int main()
{
	CenterFaceDectector m_CenterFaceDectector;
	Config m_config;
	m_config.onnxModelpath = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\CenterFace\\centerface_bnmerged.onnx";
	m_config.engineFile = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\CenterFace\\centerface_bnmerged_fp32_batch_1.engine";
	m_config.calibration_image_list_file = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\CenterFace\\image\\";
	m_config.maxBatchSize = 1;
	m_config.mode = 0;
	m_config.conf_thresh = 0.5;
	m_config.calibration_width = 640;
	m_config.calibration_height = 640;
	m_CenterFaceDectector.init(m_config);
	std::vector<BatchResult> batch_res;
	std::vector<cv::Mat> batch_img;
	std::string filename = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\CenterFace\\image\\test4.jpg";
	//cv::Mat image = cv::imread(filename);
	////test1
	//batch_img.push_back(image);
	//batch_img.push_back(image_1);

	float all_time = 0.0;
	Timer timer;
	std::vector<cv::String> m_list;
	cv::glob(m_config.calibration_image_list_file, m_list);
	int m = 100;
	for (int i = 0; i < m; i++)
	{
		cv::Mat image = cv::imread(m_list[0]);
		//test1
		batch_img.clear();
		batch_img.push_back(image);
		timer.reset();
		m_CenterFaceDectector.detect(batch_img, batch_res);
		//double t = timer.elapsed();
		double t = timer.elapsed();
		std::cout << i << ":" << t << "ms" << std::endl;
		if (i > 0)
		{
			all_time += t;
		}

		//disp
		for (int j = 0; j < batch_img.size(); ++j)
		{
			for (const auto& r : batch_res[j])
			{
				cv::rectangle(batch_img[j], r.rect, cv::Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "score:" << r.prob;
				cv::putText(batch_img[j], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
				for (int k = 0; k < r.keypoints.size(); k++)
				{
					cv::Point2f key_point = r.keypoints[k];
					if (k % 3 == 0)
						cv::circle(batch_img[j], key_point, 3, cv::Scalar(0, 255, 0), -1);
					else if (k % 3 == 1)
						cv::circle(batch_img[j], key_point, 3, cv::Scalar(0, 0, 255), -1);
					else
						cv::circle(batch_img[j], key_point, 3, cv::Scalar(0, 255, 255), -1);
				}
			}
			//cv::imshow("image" + std::to_string(i), batch_img[i]);
			//cv::imwrite("D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\CenterFace\\result\\image"  + std::to_string(i) +".png",
			//	batch_img[j]);
		}
	}
	std::cout << m << "次 time:" << all_time << " ms" << std::endl;
	std::cout << "1次 time:" << all_time / m << " ms" << std::endl;
	std::cout << "FPS::" << 1000 / (all_time / m) << std::endl;
	cv::waitKey(10);
	return 0;
}