cd ..
python train.py --img 640 --batch 8 --epochs 50 --weights weights/yolov5s_v4.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name s_hand
cd script