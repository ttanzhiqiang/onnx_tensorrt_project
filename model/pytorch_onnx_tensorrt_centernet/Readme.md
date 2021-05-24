https://drive.google.com/drive/folders/1KzBjmCOG9ghcq9L6-iqfz6QwBQq6Hl4_?usp=sharing or https://share.weiyun.com/td9CRDhW
open pytorch_onnx_tensorrt_centernet\src\centernet_to_onnx.py

#window
step1:open  x64 Native Tools Command Prompt for VS 2017 by administator
step2:open C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\Scripts\anaconda.exe
step3:cd .\src\lib\models\networks\dcn
step4:pip install torch
step5:python setup.py build_ext
step6:get deform_pool_cuda.cp36-win_amd64.pyd and deform_conv_cuda.cp36-win_amd64.pyd