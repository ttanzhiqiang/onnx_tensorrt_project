RetinaFace：https://github.com/deepinsight/insightface/tree/master/detection/retinaface

https://drive.google.com/drive/folders/1KzBjmCOG9ghcq9L6-iqfz6QwBQq6Hl4_?usp=sharing or https://share.weiyun.com/td9CRDhW
#yolor

open code\sample_retinaface_to_onnx.py

pip install onnx
pip install mxnet

pip install onnxruntime

	python3 sample_retinaface_to_onnx.py \
	                --model_symbol D:\Retinaface-TensorRT\weight\mnet.25-symbol.json
	                --model_params D:\Retinaface-TensorRT\weight\mnet.25-0000.params
	                --batch_size 1
	                --im_width 512
	                --im_height 512
	                --onnx_path D:\Retinaface-TensorRT\weight\mnet.25-512x512-batchsize_1.onnx

