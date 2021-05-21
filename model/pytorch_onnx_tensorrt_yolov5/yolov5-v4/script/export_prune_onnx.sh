cd ..
export PYTHONPATH="$PWD" && python models/export_prune_onnx.py --weights runs/train/s_hand_finetune_distill/weights/last.pt --img 640 --batch 1
cd script