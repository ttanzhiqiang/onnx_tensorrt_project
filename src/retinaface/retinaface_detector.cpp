#include "assert.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <common.h>
#include "Trt.h"
#include "class_timer.hpp"
struct FacePts
{
	float x[5];
	float y[5];
};

struct RetinaFaceResult
{
	float	 prob = 0.f;
	cv::Rect rect;
	FacePts m_FacePts;
};
typedef std::vector<RetinaFaceResult> BatchResult;

struct anchor_cfg
{
public:
	int STRIDE;
	std::vector<int> SCALES;
	int BASE_SIZE;
	std::vector<float> RATIOS;
	int ALLOWED_BORDER;

	anchor_cfg()
	{
		STRIDE = 0;
		SCALES.clear();
		BASE_SIZE = 0;
		RATIOS.clear();
		ALLOWED_BORDER = 0;
	}
};

struct anchor_box
{
	float x1;
	float y1;
	float x2;
	float y2;
};

struct anchor_win
{
	float x_ctr;
	float y_ctr;
	float w;
	float h;
};

struct FaceDetectInfo
{
	float score;
	anchor_box rect;
	FacePts pts;
};

class RetinaFaceDectector
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
	float conf_thresh = 0.6;
	int _n_yolo_ind = 0;
	std::vector<TensorInfo> m_OutputTensors;
	std::vector<std::map<std::string, std::string>> m_configBlocks;
	cudaStream_t mCudaStream;
	Config _config;
	std::vector<int> _feat_stride_fpn = { 32, 16, 8 };
	std::vector<anchor_cfg> cfg;
	std::vector<float> _ratio = { 1.0};
	std::map<std::string, std::vector<anchor_box>> _anchors_fpn;
	std::map<std::string, int> _num_anchors;
	std::map<std::string, std::vector<anchor_box>> _anchors;
public:
	RetinaFaceDectector::RetinaFaceDectector()
	{

	}
	RetinaFaceDectector::~RetinaFaceDectector()
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

	std::vector<anchor_box> anchors_plane(int height, int width, int stride, std::vector<anchor_box> base_anchors)
	{
		/*
		height: height of plane
		width:  width of plane
		stride: stride ot the original image
		anchors_base: a base set of anchors
		*/

		std::vector<anchor_box> all_anchors;
		for (size_t k = 0; k < base_anchors.size(); k++) {
			for (int ih = 0; ih < height; ih++) {
				int sh = ih * stride;
				for (int iw = 0; iw < width; iw++) {
					int sw = iw * stride;

					anchor_box tmp;
					tmp.x1 = base_anchors[k].x1 + sw;
					tmp.y1 = base_anchors[k].y1 + sh;
					tmp.x2 = base_anchors[k].x2 + sw;
					tmp.y2 = base_anchors[k].y2 + sh;
					all_anchors.push_back(tmp);
				}
			}
		}

		return all_anchors;
	}


	TensorInfo getOutputTensors(std::string name)
	{
		TensorInfo m_TensorInfo;
		for (int i = 0; i < m_OutputTensors.size(); i++)
		{
			if (strcmp(m_OutputTensors[i].blobName.c_str(),name.c_str()) == 0)
			{
				m_TensorInfo = m_OutputTensors[i];
				break;
			}
		}
		return m_TensorInfo;
	}

	anchor_box _mkanchors(anchor_win win)
	{
		//Given a vector of widths (ws) and heights (hs) around a center
		//(x_ctr, y_ctr), output a set of anchors (windows).
		anchor_box anchor;
		anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
		anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
		anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
		anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

		return anchor;
	}

	anchor_win  _whctrs(anchor_box anchor)
	{
		//Return width, height, x center, and y center for an anchor (window).
		anchor_win win;
		win.w = anchor.x2 - anchor.x1 + 1;
		win.h = anchor.y2 - anchor.y1 + 1;
		win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
		win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

		return win;
	}

	std::vector<anchor_box> _ratio_enum(anchor_box anchor, std::vector<float> ratios)
	{
		//Enumerate a set of anchors for each aspect ratio wrt an anchor.
		std::vector<anchor_box> anchors;
		for (size_t i = 0; i < ratios.size(); i++) {
			anchor_win win = _whctrs(anchor);
			float size = win.w * win.h;
			float scale = size / ratios[i];

			win.w = std::round(sqrt(scale));
			win.h = std::round(win.w * ratios[i]);

			anchor_box tmp = _mkanchors(win);
			anchors.push_back(tmp);
		}

		return anchors;
	}

	std::vector<anchor_box> _scale_enum(anchor_box anchor, std::vector<int> scales)
	{
		//Enumerate a set of anchors for each scale wrt an anchor.
		std::vector<anchor_box> anchors;
		for (size_t i = 0; i < scales.size(); i++) {
			anchor_win win = _whctrs(anchor);

			win.w = win.w * scales[i];
			win.h = win.h * scales[i];

			anchor_box tmp = _mkanchors(win);
			anchors.push_back(tmp);
		}

		return anchors;
	}

	std::vector<anchor_box> generate_anchors(int base_size = 16, std::vector<float> ratios = { 0.5, 1, 2 },
		std::vector<int> scales = { 8, 64 }, int stride = 16, bool dense_anchor = false)
	{
		//Generate anchor (reference) windows by enumerating aspect ratios X
		//scales wrt a reference (0, 0, 15, 15) window.

		anchor_box base_anchor;
		base_anchor.x1 = 0;
		base_anchor.y1 = 0;
		base_anchor.x2 = base_size - 1;
		base_anchor.y2 = base_size - 1;

		std::vector<anchor_box> ratio_anchors;
		ratio_anchors = _ratio_enum(base_anchor, ratios);

		std::vector<anchor_box> anchors;
		for (size_t i = 0; i < ratio_anchors.size(); i++) {
			std::vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
			anchors.insert(anchors.end(), tmp.begin(), tmp.end());
		}

		if (dense_anchor) {
			assert(stride % 2 == 0);
			std::vector<anchor_box> anchors2 = anchors;
			for (size_t i = 0; i < anchors2.size(); i++) {
				anchors2[i].x1 += stride / 2;
				anchors2[i].y1 += stride / 2;
				anchors2[i].x2 += stride / 2;
				anchors2[i].y2 += stride / 2;
			}
			anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
		}

		return anchors;
	}

	std::vector<std::vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, std::vector<anchor_cfg> cfg = {})
	{
		//Generate anchor (reference) windows by enumerating aspect ratios X
		//scales wrt a reference (0, 0, 15, 15) window.

		std::vector<std::vector<anchor_box>> anchors;
		for (size_t i = 0; i < cfg.size(); i++) {
			//stride从小到大[32 16 8]
			anchor_cfg tmp = cfg[i];
			int bs = tmp.BASE_SIZE;
			std::vector<float> ratios = tmp.RATIOS;
			std::vector<int> scales = tmp.SCALES;
			int stride = tmp.STRIDE;

			std::vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
			anchors.push_back(r);
		}

		return anchors;
	}

	void clip_boxes(anchor_box& box, int width, int height)
	{
		//Clip boxes to image boundaries.
		if (box.x1 < 0) {
			box.x1 = 0;
		}
		if (box.y1 < 0) {
			box.y1 = 0;
		}
		if (box.x2 > width - 1) {
			box.x2 = width - 1;
		}
		if (box.y2 > height - 1) {
			box.y2 = height - 1;
		}
	}

	anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress)
	{
		anchor_box rect;

		float width = anchor.x2 - anchor.x1 + 1;
		float height = anchor.y2 - anchor.y1 + 1;
		float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
		float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

		float pred_ctr_x = regress[0] * width + ctr_x;
		float pred_ctr_y = regress[1] * height + ctr_y;
		float pred_w = exp(regress[2]) * width;
		float pred_h = exp(regress[3]) * height;

		rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
		rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
		rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
		rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

		return rect;
	}

	FacePts landmark_pred(anchor_box anchor, FacePts facePt)
	{
		FacePts pt;
		float width = anchor.x2 - anchor.x1 + 1;
		float height = anchor.y2 - anchor.y1 + 1;
		float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
		float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

		for (size_t j = 0; j < 5; j++) {
			pt.x[j] = facePt.x[j] * width + ctr_x;
			pt.y[j] = facePt.y[j] * height + ctr_y;
		}

		return pt;
	}

	static bool CompareBBox(const FaceDetectInfo& a, const FaceDetectInfo& b)
	{
		return a.score > b.score;
	}

	std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo>& bboxes, float threshold)
	{
		std::vector<FaceDetectInfo> bboxes_nms;
		std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

		int32_t select_idx = 0;
		int32_t num_bbox = static_cast<int32_t>(bboxes.size());
		std::vector<int32_t> mask_merged(num_bbox, 0);
		bool all_merged = false;

		while (!all_merged) {
			while (select_idx < num_bbox && mask_merged[select_idx] == 1)
				select_idx++;
			//如果全部执行完则返回
			if (select_idx == num_bbox) {
				all_merged = true;
				continue;
			}

			bboxes_nms.push_back(bboxes[select_idx]);
			mask_merged[select_idx] = 1;

			anchor_box select_bbox = bboxes[select_idx].rect;
			float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
			float x1 = static_cast<float>(select_bbox.x1);
			float y1 = static_cast<float>(select_bbox.y1);
			float x2 = static_cast<float>(select_bbox.x2);
			float y2 = static_cast<float>(select_bbox.y2);

			select_idx++;
			for (int32_t i = select_idx; i < num_bbox; i++) {
				if (mask_merged[i] == 1)
					continue;

				anchor_box& bbox_i = bboxes[i].rect;
				float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
				float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
				float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
				float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
				if (w <= 0 || h <= 0)
					continue;

				float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
				float area_intersect = w * h;


				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
					mask_merged[i] = 1;
				}
			}
		}

		return bboxes_nms;
	}

	std::vector<FaceDetectInfo> decodeTensor(const int imageIdx,
		const int imageH,
		const int imageW,
		const int _feat_stride_fpn_i)
	{
		std::string name_bbox = "face_rpn_bbox_pred_stride";
		std::string name_score = "face_rpn_cls_prob_reshape_stride";
		std::string name_landmark = "face_rpn_landmark_pred_stride";
		std::string key = "stride" + std::to_string(_feat_stride_fpn_i);
		std::string str_name_bbox = name_bbox + std::to_string(_feat_stride_fpn_i);
		std::string str_name_score = name_score + std::to_string(_feat_stride_fpn_i);
		std::string str_name_landmark = name_landmark + std::to_string(_feat_stride_fpn_i);
		auto tensor_bbox = getOutputTensors(str_name_bbox);
		auto tensor_score = getOutputTensors(str_name_score);
		auto tensor_landmark = getOutputTensors(str_name_landmark);

		size_t num_anchor = _num_anchors[key];
		size_t count = tensor_score.grid_w * tensor_score.grid_h;
		std::vector<FaceDetectInfo> faceInfo;
		//存储顺序 h * w * num_anchor
		std::vector<anchor_box> anchors = anchors_plane(tensor_score.grid_h, tensor_score.grid_w,
			_feat_stride_fpn_i, _anchors_fpn[key]);
		for (size_t num = 0; num < num_anchor; num++) {
			for (size_t j = 0; j < count; j++) {
				//置信度小于阈值跳过
				int half_num = tensor_score.hostBuffer.size() / 2;
				float conf = tensor_score.hostBuffer[j + count * num + half_num];
				if (conf <= conf_thresh) {
					continue;
				}

				cv::Vec4f regress;
				float dx = tensor_bbox.hostBuffer[j + count * (0 + num * 4)];
				float dy = tensor_bbox.hostBuffer[j + count * (1 + num * 4)];
				float dw = tensor_bbox.hostBuffer[j + count * (2 + num * 4)];
				float dh = tensor_bbox.hostBuffer[j + count * (3 + num * 4)];
				regress = cv::Vec4f(dx, dy, dw, dh);

				//回归人脸框
				anchor_box rect = bbox_pred(anchors[j + count * num], regress);
				//越界处理
				clip_boxes(rect, m_InputW, m_InputH);

				rect.x1 = rect.x1 / m_InputW * imageW;
				rect.x2 = rect.x2 / m_InputW * imageW;
				rect.y1 = rect.y1 / m_InputH * imageH;
				rect.y2 = rect.y2 / m_InputH * imageH;
				FacePts pts;
				for (size_t k = 0; k < 5; k++) {
					pts.x[k] = tensor_landmark.hostBuffer[j + count * (num * 10 + k * 2)];
					pts.y[k] = tensor_landmark.hostBuffer[j + count * (num * 10 + k * 2 + 1)];
				}
				//回归人脸关键点
				FacePts landmarks = landmark_pred(anchors[j + count * num], pts);
				for (int i = 0; i < 5; i++)
				{
					landmarks.x[i] = landmarks.x[i] / m_InputW * imageW;
					landmarks.y[i] = landmarks.y[i] / m_InputH * imageH;
				}

				FaceDetectInfo tmp;
				tmp.score = conf;
				tmp.rect = rect;
				tmp.pts = landmarks;
				faceInfo.push_back(tmp);
			}
		}
		return faceInfo;
	}
	std::vector<FaceDetectInfo> decodeDetections(const int& imageIdx,
		const int& imageH,
		const int& imageW)
	{
		std::vector<FaceDetectInfo> faceInfo;
		for (int m_tensor_i = 0; m_tensor_i < _feat_stride_fpn.size(); m_tensor_i++)
		{
			std::vector<FaceDetectInfo> curBInfo = decodeTensor(imageIdx, imageH, imageW, _feat_stride_fpn[m_tensor_i]);
			faceInfo.insert(faceInfo.end(), curBInfo.begin(), curBInfo.end());
		}
		//排序nms
		faceInfo = nms(faceInfo, m_NMSThresh);
		return faceInfo;
	}

	void UpdateOutputTensor()
	{
		m_InputC = onnx_net->mBindingDims[0].d[1];
		m_InputW = onnx_net->mBindingDims[0].d[2];
		m_InputH = onnx_net->mBindingDims[0].d[3];
		m_Classes = onnx_net->mBindingDims[1].d[2] - 5;
		for (int m_yolo_ind = 1; m_yolo_ind < onnx_net->mBindingName.size(); m_yolo_ind++)
		{
			TensorInfo outputTensor;
			outputTensor.grid_w = onnx_net->mBindingDims[m_yolo_ind].d[2];
			outputTensor.grid_h = onnx_net->mBindingDims[m_yolo_ind].d[3];
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

	void init_anchor()
	{
		anchor_cfg tmp;
		tmp.SCALES = { 32, 16 };
		tmp.BASE_SIZE = 16;
		tmp.RATIOS = _ratio;
		tmp.ALLOWED_BORDER = 9999;
		tmp.STRIDE = 32;
		cfg.push_back(tmp);

		tmp.SCALES = { 8, 4 };
		tmp.BASE_SIZE = 16;
		tmp.RATIOS = _ratio;
		tmp.ALLOWED_BORDER = 9999;
		tmp.STRIDE = 16;
		cfg.push_back(tmp);

		tmp.SCALES = { 2, 1 };
		tmp.BASE_SIZE = 16;
		tmp.RATIOS = _ratio;
		tmp.ALLOWED_BORDER = 9999;
		tmp.STRIDE = 8;
		cfg.push_back(tmp);
		bool dense_anchor = false;
		std::vector<std::vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
		std::vector<int> outputH;
		std::vector<int> outputW;
		for (int i = 0; i < _feat_stride_fpn.size(); i++)
		{
			outputH.push_back(m_InputH / _feat_stride_fpn[i]);
			outputW.push_back(m_InputW / _feat_stride_fpn[i]);
		}


		for (size_t i = 0; i < anchors_fpn.size(); i++) {
			int stride = _feat_stride_fpn[i];
			std::string key = "stride" + std::to_string(_feat_stride_fpn[i]);
			_anchors_fpn[key] = anchors_fpn[i];
			_num_anchors[key] = anchors_fpn[i].size();
			//有三组不同输出宽高
			_anchors[key] = anchors_plane(outputH[i], outputW[i], stride, _anchors_fpn[key]);
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
		conf_thresh = config.conf_thresh;
		m_BatchSize = config.maxBatchSize;
		onnx_net->CreateEngine(config.onnxModelpath, config.engineFile, config.customOutput, config.maxBatchSize, config.mode);
		//更新m_OutputTensors
		UpdateOutputTensor();
		init_anchor();
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
			//float ratio = float(m_InputW) / float(src_img.cols) < float(m_InputH) / float(src_img.rows) ? float(m_InputW) / float(src_img.cols) : float(m_InputH) / float(src_img.rows);
			float ratio_w = float(m_InputW) / float(src_img.cols);
			float ratio_h = float(m_InputH) / float(src_img.rows);
			cv::Mat flt_img = cv::Mat::zeros(cv::Size(m_InputW, m_InputH), CV_8UC3);
			cv::Mat rsz_img;
			cv::resize(src_img, rsz_img, cv::Size(), ratio_w, ratio_h);
			rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
			flt_img.convertTo(flt_img, CV_32FC3);

			//HWC TO CHW
			int channelLength = m_InputW * m_InputH;
			std::vector<cv::Mat> split_img = {
					cv::Mat(m_InputW, m_InputW, CV_32FC1, data + channelLength * (index + 2)),
					cv::Mat(m_InputW, m_InputW, CV_32FC1, data + channelLength * (index + 1)),
					cv::Mat(m_InputW, m_InputW, CV_32FC1, data + channelLength * index)
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
		std::vector<float>data;
		data = prepareImage(vec_image);
		doInference(data, vec_image.size());
		for (uint32_t i = 0; i < vec_image.size(); ++i)
		{
			auto curImage = vec_image.at(i);
			auto binfo = decodeDetections(i, curImage.rows, curImage.cols);
			if (binfo.empty())
			{
				continue;
			}
			std::vector<RetinaFaceResult> vec_result;
			for (const auto& b : binfo)
			{
				RetinaFaceResult res;
				res.prob = b.score;
				const int x = b.rect.x1;
				const int y = b.rect.y1;
				const int w = b.rect.x2 - b.rect.x1;
				const int h = b.rect.y2 - b.rect.y1;
				res.rect = cv::Rect(x, y, w, h);
				res.m_FacePts = b.pts;
				vec_result.push_back(res);
			}
			vec_batch_result.push_back(vec_result);
		}
	}
};

int main()
{
	RetinaFaceDectector m_RetinaFaceDectector;
	Config m_config;
	m_config.onnxModelpath = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\mxnet_onnx_tensorrt_retinaface\\mnet.25-512x512-batchsize_1.onnx";
	m_config.engineFile = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\mxnet_onnx_tensorrt_retinaface\\mnet.25-512x512-int8_batchsize_1.engine";
	m_config.calibration_image_list_file = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\mxnet_onnx_tensorrt_retinaface\\image\\";
	m_config.maxBatchSize = 1;
	m_config.mode = 2;
	m_config.calibration_width = 512;
	m_config.calibration_height = 512;
	m_config.conf_thresh = 0.2;
	m_RetinaFaceDectector.init(m_config);
	std::vector<BatchResult> batch_res;
	std::vector<cv::Mat> batch_img;
	std::string filename = "D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\mxnet_onnx_tensorrt_retinaface\\image\\lumia.jpg";
	cv::Mat image = cv::imread(filename);
	//test1
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
		m_RetinaFaceDectector.detect(batch_img, batch_res);
		//double t = timer.elapsed();
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
			std::cout << "batch " << i << " prob:" << r.prob << " rect:" << r.rect << std::endl;
			cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2)  << "  score:" << r.prob;
			cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
			for (size_t j = 0; j < 5; j++) {
				cv::Point2f pt = cv::Point2f(r.m_FacePts.x[j], r.m_FacePts.y[j]);
				cv::circle(batch_img[i], pt, 1, cv::Scalar(0, 255, 0), 2);
			}
		}
		cv::imshow("image" + std::to_string(i), batch_img[i]);
		//cv::imwrite("D:\\onnx_tensorrt\\onnx_tensorrt_centernet\\onnx_tensorrt_project\\model\\mxnet_onnx_tensorrt_retinaface\\result\\image"  + std::to_string(i) +".png",batch_img[i]);
	}
	cv::waitKey(10);
	return 0;
}