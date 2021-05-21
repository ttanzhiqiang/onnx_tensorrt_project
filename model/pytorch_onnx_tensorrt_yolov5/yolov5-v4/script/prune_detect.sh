cd ..
python prune_detect.py --weights weights/last_s_hand_finetune.pt --img  640 --conf 0.7 --save-txt --source /home/lishuang/Disk/gitlab/traincode/yolov5/data/hand_dataset/images/test
cd script