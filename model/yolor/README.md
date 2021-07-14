https://drive.google.com/drive/folders/1KzBjmCOG9ghcq9L6-iqfz6QwBQq6Hl4_?usp=sharing or https://share.weiyun.com/td9CRDhW
#yolor

open code\convert_to_onnx.py

pip install onnx
pip install torch

python convert_to_onnx.py --weights yolor_csp_x_star.pt --cfg cfg/yolor_csp_x.cfg --output yolo_csp_x_star.onnx
