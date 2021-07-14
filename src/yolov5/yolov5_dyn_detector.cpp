#include "assert.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <common.h>
#include "Trt.h"
struct Result
{
	int		 id = -1;
	float	 prob = 0.f;
	cv::Rect rect;
};
typedef std::vector<Result> BatchResult;
class Yolov5DynDectector
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
	int _n_yolo_ind = 0;
	std::vector<TensorInfo> m_OutputTensors;
	std::vector<std::map<std::string, std::string>> m_configBlocks;
	cudaStream_t mCudaStream;
	Config _config;
	std::vector<float> vec_anchors = { 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 };

	uint32_t m_Ori_InputH;
	uint32_t m_Ori_InputW;
public:
	Yolov5DynDectector::Yolov5DynDectector()
	{

	}
	Yolov5DynDectector::~Yolov5DynDectector()
	{
		//释放内存
		for (int i = 0; i < m_OutputTensors.size();i++)
		{
			m_OutputTensors[i].hostBuffer.clear();
		}
		m_OutputTensors.clear();
		m_configBlocks.clear();
		cudaStreamDestroy(mCudaStream);
	}
	const std::vector<int> m_ClassIds{
		1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
		22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
		46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
		67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };

	int getClassId(const int& label) { return m_ClassIds.at(label); }

	float clamp(const float val, const float minVal, const float maxVal)
	{
		assert(minVal <= maxVal);
		return std::min(maxVal, std::max(minVal, val));
	}

	BBox convert_bbox_res(const float& bx, const float& by, const float& bw, const float& bh,
		const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH)
	{
		BBox b;
		// Restore coordinates to network input resolution
		float x = bx * stride_w_;
		float y = by * stride_h_;

		b.x1 = x - bw / 2;
		b.x2 = x + bw / 2;

		b.y1 = y - bh / 2;
		b.y2 = y + bh / 2;

		b.x1 = clamp(b.x1, 0, netW);
		b.x2 = clamp(b.x2, 0, netW);
		b.y1 = clamp(b.y1, 0, netH);
		b.y2 = clamp(b.y2, 0, netH);

		return b;
	}

	inline void add_bbox_proposal(const float bx, const float by, const float bw, const float bh,
		const uint32_t stride_h_, const uint32_t stride_w_, const float scaleH, const float scaleW, const float xoffset_, const float yoffset, const int maxIndex, const float maxProb,
		const uint32_t 	image_w, const uint32_t image_h,
		std::vector<BBoxInfo>& binfo)
	{
		BBoxInfo bbi;
		bbi.box = convert_bbox_res(bx, by, bw, bh, stride_h_, stride_w_, m_InputW, m_InputH);
		if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
		{
			return;
		}
		else
		{
			bbi.box.x1 = ((float)bbi.box.x1 / (float)m_InputW) * (float)image_w;
			bbi.box.y1 = ((float)bbi.box.y1 / (float)m_InputH) * (float)image_h;
			bbi.box.x2 = ((float)bbi.box.x2 / (float)m_InputW) * (float)image_w;
			bbi.box.y2 = ((float)bbi.box.y2 / (float)m_InputH) * (float)image_h;
		}

		bbi.label = maxIndex;
		bbi.prob = maxProb;
		bbi.classId = getClassId(maxIndex);
		binfo.push_back(bbi);
	};

	void calcuate_letterbox_message(const int m_InputH, const int m_InputW,
		const int imageH, const int imageW,
		float& sh, float& sw,
		int& xOffset, int& yOffset)
	{
		float dim = std::max(imageW, imageH);
		int resizeH = ((imageH / dim) * m_InputH);
		int resizeW = ((imageW / dim) * m_InputW);
		sh = static_cast<float>(resizeH) / static_cast<float>(imageH);
		sw = static_cast<float>(resizeW) / static_cast<float>(imageW);
		if ((m_InputW - resizeW) % 2) resizeW--;
		if ((m_InputH - resizeH) % 2) resizeH--;
		assert((m_InputW - resizeW) % 2 == 0);
		assert((m_InputH - resizeH) % 2 == 0);
		xOffset = (m_InputW - resizeW) / 2;
		yOffset = (m_InputH - resizeH) / 2;
	}

	std::vector<BBoxInfo> decodeTensor(const int imageIdx,
		const int imageH,
		const int imageW,
		const int m_tensor_i)
	{
		auto tensor = m_OutputTensors[m_tensor_i];
		float* detections = tensor.hostBuffer.data() + imageIdx * tensor.volume;
		std::vector<BBoxInfo> binfo;
		int position = 0;
		for (int c = 0; c < tensor.numBBoxes; c++)
		{
			const float pw = tensor.anchors[tensor.masks[c] * 2];
			const float ph = tensor.anchors[tensor.masks[c] * 2 + 1];
			for (int h = 0; h < tensor.grid_h; h++)
			{
				for (int w = 0; w < tensor.grid_w; w++)
				{
					float* row = detections + position * (m_Classes + 5);
					position++;
					BBoxInfo box;
					auto max_pos = std::max_element(row + 5, row + m_Classes + 5);
					box.prob = Logist(row[4]) * Logist(row[max_pos - row]);
					if (box.prob < 0.5)
						continue;
					box.classId = max_pos - row - 5;
					box.label = max_pos - row - 5;
					int center_x = (Logist(row[0]) * 2 - 0.5 + w) * tensor.stride_w;
					int center_y = (Logist(row[1]) * 2 - 0.5 + h) * tensor.stride_h;
					int center_w = pow(Logist(row[2]) * 2.f, 2) * pw;
					int center_h = pow(Logist(row[3]) * 2.f, 2) * ph;
					box.box.x1 = double(center_x - center_w / 2) / m_InputW * imageW;
					box.box.x2 = double(center_x + center_w / 2) / m_InputW * imageW;
					box.box.y1 = double(center_y - center_h / 2) / m_InputH * imageH;
					box.box.y2 = double(center_y + center_h / 2) / m_InputH * imageH;
					binfo.push_back(box);
				}
			}
		}
		return binfo;
	}

	std::vector<BBoxInfo> decodeDetections(const int& imageIdx,
		const int& imageH,
		const int& imageW)
	{
		//	Timer timer;
		std::vector<BBoxInfo> binfo;
		for (int m_tensor_i = 0; m_tensor_i< m_OutputTensors.size(); m_tensor_i++)
		{
			std::vector<BBoxInfo> curBInfo = decodeTensor(imageIdx, imageH, imageW, m_tensor_i);
			binfo.insert(binfo.end(), curBInfo.begin(), curBInfo.end());
		}
		//	timer.out("decodeDetections");
		return binfo;
	}


	__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };	

	std::vector<BBoxInfo> diou_nms(const float nmsThresh, std::vector<BBoxInfo> binfo)
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

		//https://arxiv.org/pdf/1911.08287.pdf
		auto R = [](BBox& bbox1, BBox& bbox2) ->float
		{
			float center1_x = (bbox1.x1 + bbox1.x2) / 2.f;
			float center1_y = (bbox1.y1 + bbox1.y2) / 2.f;
			float center2_x = (bbox2.x1 + bbox2.x2) / 2.f;
			float center2_y = (bbox2.y1 + bbox2.y2) / 2.f;

			float d_center = (center1_x - center2_x) * (center1_x - center2_x)
				+ (center1_y - center2_y) * (center1_y - center2_y);
			//smallest_enclosing box
			float box_x1 = std::min({ bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2 });
			float box_y1 = std::min({ bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2 });
			float box_x2 = std::max({ bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2 });
			float box_y2 = std::max({ bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2 });

			float d_diagonal = (box_x1 - box_x2) * (box_x1 - box_x2) +
				(box_y1 - box_y2) * (box_y1 - box_y2);

			return d_center / d_diagonal;
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
					float r = R(i.box, j.box);
					keep = (overlap - r) <= nmsThresh;
				}
				else
					break;
			}
			if (keep) out.push_back(i);
		}
		return out;
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
			std::string model_type = "yolov5";
			if ("yolov5" == model_type)
			{
				boxes = diou_nms(nmsThresh, boxes);
			}
			else
			{
				boxes = nonMaximumSuppression(nmsThresh, boxes);
			}
			result.insert(result.end(), boxes.begin(), boxes.end());
		}

		return result;
	}

	float getNMSThresh() const { return m_NMSThresh; }

	void UpdateOutputTensor()
	{
		m_InputW = onnx_net->mBindingDims[0].d[2];
		m_InputH = onnx_net->mBindingDims[0].d[3];
		m_Classes = onnx_net->mBindingDims[1].d[4] - 5;
		for (int m_yolo_ind = 1; m_yolo_ind < onnx_net->mBindingName.size(); m_yolo_ind++)
		{
			TensorInfo outputTensor;
			if (m_yolo_ind < 4)
			{
				outputTensor.anchors = vec_anchors;
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

	void isInt8(std::string calibration_image_list_file)
	{
		size_t npos = _config.onnxModelpath.find_first_of('.');
		std::string calib_table_name = _config.onnxModelpath.substr(0, npos) + ".table";
		if (!fileExists(calib_table_name))
		{
			onnx_net->SetInt8Calibrator("Int8MinMaxCalibrator", m_InputW, m_InputH, calibration_image_list_file.c_str(), calib_table_name.c_str());
		}
	}

	void init(Config config)
	{
		_config = config;
		onnx_net = std::make_shared<Trt>();
		onnx_net->SetMaxBatchSize(config.maxBatchSize);
		if (config.mode == 2)
		{
			isInt8(config.calibration_image_list_file);
		}
		bool m_dyn_flag = true;
		if (m_dyn_flag)
		{
			onnx_net->AddDynamicShapeProfile(1, "input", { 3,320,320 }, { 3,512,512 }, { 3,800,800 });
		}
		m_BatchSize = config.maxBatchSize;
		onnx_net->CreateEngine(config.onnxModelpath, config.engineFile, config.customOutput, config.maxBatchSize, config.mode);
		onnx_net->buildPreprocessorEngine();
		//更新m_OutputTensors
		UpdateOutputTensor();
		allocateBuffers();
		cudaStreamCreate(&mCudaStream);
	}

	void doInference_dyn(std::vector<float> input, const uint32_t batchSize)
	{
		//	Timer timer;
		//assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
		//onnx_net->CopyFromHostToDevice(input, 0, mCudaStream);

		nvinfer1::Dims inputdim = onnx_net->mEngine->getBindingDimensions(0); // C*H*W
		void* hostBuffer;
		cudaMalloc(&hostBuffer, input.size() * sizeof(float));
		CUDA_CHECK(cudaMemcpy(hostBuffer, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

		// Set the input size for the preprocessor
		nvinfer1::Dims inputDims_1;
		inputDims_1.nbDims = 4;
		inputDims_1.d[0] = batchSize;
		inputDims_1.d[1] = m_InputC;
		inputDims_1.d[2] = m_Ori_InputH;
		inputDims_1.d[3] = m_Ori_InputW;
		//CHECK_RETURN_W_MSG(onnx_net->mPreprocessorContext->setBindingDimensions(0, inputDims), false, "Invalid binding dimensions.");
		onnx_net->mPreprocessorContext->setBindingDimensions(0, inputDims_1);
		// We can only run inference once all dynamic input shapes have been specified.
		if (!onnx_net->mPreprocessorContext->allInputDimensionsSpecified())
		{
			return;
		}

		std::vector<void*> preprocessorBindings = { hostBuffer, onnx_net->GetBindingPtr(0) };
		bool status = onnx_net->mPreprocessorContext->executeV2(preprocessorBindings.data());
		if (!status)
		{
			return;
		}

		onnx_net->ForwardAsync(mCudaStream);
		for (auto& tensor : m_OutputTensors)
		{
			onnx_net->CopyFromDeviceToHost(tensor.hostBuffer, tensor.bindingIndex, mCudaStream);
		}
		cudaStreamSynchronize(mCudaStream);
	}

	void detect_dyn(const std::vector<cv::Mat>& vec_image,
		std::vector<BatchResult>& vec_batch_result)
	{
		m_Ori_InputW = vec_image[0].cols;
		m_Ori_InputH = vec_image[0].rows;
		m_InputC = vec_image[0].channels();
		vec_batch_result.clear();
		vec_batch_result.resize(vec_image.size());
		std::vector<float>data;
		for (const auto& img : vec_image)
		{
			cv::Mat imgf;
			img.convertTo(imgf, CV_32FC3, 1 / 255.0);
			std::vector<cv::Mat>channles(3);
			cv::split(imgf, channles);
			float* ptr1 = (float*)(channles[0].data);
			float* ptr2 = (float*)(channles[1].data);
			float* ptr3 = (float*)(channles[2].data);
			data.insert(data.end(), ptr1, ptr1 + img.rows * img.cols);
			data.insert(data.end(), ptr2, ptr2 + img.rows * img.cols);
			data.insert(data.end(), ptr3, ptr3 + img.rows * img.cols);
		}

		doInference_dyn(data, vec_image.size());
		for (uint32_t i = 0; i < vec_image.size(); ++i)
		{
			auto curImage = vec_image.at(i);
			auto binfo = decodeDetections(i, curImage.rows, curImage.cols);
			auto remaining = nmsAllClasses(getNMSThresh(),
				binfo,
				m_Classes,
				"");
			if (remaining.empty())
			{
				continue;
			}
			std::vector<Result> vec_result(0);
			for (const auto& b : remaining)
			{
				Result res;
				res.id = b.label;
				res.prob = b.prob;
				const int x = b.box.x1;
				const int y = b.box.y1;
				const int w = b.box.x2 - b.box.x1;
				const int h = b.box.y2 - b.box.y1;
				res.rect = cv::Rect(x, y, w, h);
				vec_result.push_back(res);
			}
			vec_batch_result[i] = vec_result;
		}
	}
};

int main()
{
	Yolov5DynDectector m_Yolov5DynDectector;
	Config m_config;
	m_config.onnxModelpath = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\yolov5\\yolov5-v5_weight\\yolov5m.onnx";
	m_config.engineFile = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\yolov5\\yolov5-v5_weight\\yolov5m_fp32_dyn_batch_1.engine";
	//m_config.calibration_image_list_file = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\darknet_onnx_tensorrt_yolo\\image\\";
	m_config.maxBatchSize = 1;
	m_config.mode = 0;
	m_Yolov5DynDectector.init(m_config);
	std::vector<BatchResult> batch_res;
	std::vector<cv::Mat> batch_img;
	std::string filename = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\darknet_onnx_tensorrt_yolo\\image\\dog.jpg";
	cv::Mat image = cv::imread(filename);
	std::string filename_1 = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\darknet_onnx_tensorrt_yolo\\image\\person.jpg";
	cv::Mat image_1 = cv::imread(filename_1);
	//test1
	batch_img.push_back(image);
	//batch_img.push_back(image_1);
	m_Yolov5DynDectector.detect_dyn(batch_img, batch_res);
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