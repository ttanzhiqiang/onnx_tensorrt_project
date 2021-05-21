python train_sparsity.py --img 416 --batch 16 --epochs 5 --data data/coco_fangweisui.yaml --cfg models/yolov5s_fangweisui.yaml --weights weights/last.pt --name s -sr --s 0.001 --prune 1

python train_sparsity.py --img 640 --batch 8 --epochs 50 --data data/coco_hand.yaml --cfg models/yolov5s.yaml --weights runs/train/s_hand/weights/last.pt --name s_hand_sparsity -sr --s 0.001 --prune 1

