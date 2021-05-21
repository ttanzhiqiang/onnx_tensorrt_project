#yolov5-4.0

open yolov5-v4\models\export_onnx.py

pip install onnx
pip install torch

--weights ./yolov5x.pt --img-size 640 --batch-size 1


#yolov5-4.0
open yolov5-v4\models\export.py

--weights ./yolov5x.pt --img 640 --batch 1 --simplify