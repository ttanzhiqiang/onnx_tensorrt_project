# python fangweisui_test.py \
# --weights ./weights/last_v4tiny.pt \
# --source /home/lishuang/Disk/shengshi_data/anti_tail_test_dataset/Data_of_each_scene \
# --output /home/lishuang/Disk/remote/pycharm/yolov4_tiny_416_04 \
# --conf-thres 0.4
# python fangweisui_test.py \
# --weights ./weights/last_v4tiny.pt \
# --source /home/lishuang/Disk/shengshi_data/anti_tail_test_dataset/double_company \
# --output /home/lishuang/Disk/remote/pycharm/yolov4_tiny_416_04 \
# --conf-thres 0.4

python fangweisui_test.py \
--weights ./weights/lastv4_eiou.pt \
--source /home/lishuang/Disk/shengshi_data/anti_tail_test_dataset/Data_of_each_scene \
--output /home/lishuang/Disk/remote/pycharm/yolov4_eiou_416_04 \
--conf-thres 0.4
python fangweisui_test.py \
--weights ./weights/lastv4_eiou.pt \
--source /home/lishuang/Disk/shengshi_data/anti_tail_test_dataset/double_company \
--output /home/lishuang/Disk/remote/pycharm/yolov4_eiou_416_04 \
--conf-thres 0.4