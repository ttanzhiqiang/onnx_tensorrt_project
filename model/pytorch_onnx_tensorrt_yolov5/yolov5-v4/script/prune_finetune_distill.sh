cd ..
python prune_finetune.py --img 640 --batch 6 --epochs 50 --data data/coco_hand.yaml --cfg ./cfg/prune_0.5_keep_0.01_8x_yolov5s_v4_hand.cfg --weights ./weights/prune_0.5_keep_0.01_8x_last_v4s.pt --name s_hand_finetune_distill --distill
cd script