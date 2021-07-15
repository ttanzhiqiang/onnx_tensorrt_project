yolov5:https://github.com/ultralytics/yolov5

https://drive.google.com/drive/folders/1KzBjmCOG9ghcq9L6-iqfz6QwBQq6Hl4_?usp=sharing or https://share.weiyun.com/td9CRDhW
#yolov5-4.0

open yolov5-v4\models\export_onnx.py

pip install onnx
pip install torch

--weights ./yolov5x.pt --img-size 640 --batch-size 1


#yolov5-4.0
open yolov5-v4\models\export.py

--weights ./yolov5x.pt --img 640 --batch 1 --simplify