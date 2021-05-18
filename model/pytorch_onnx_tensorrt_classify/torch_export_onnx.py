import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 3, 224, 224).to(device)
model = torchvision.models.resnet18(pretrained=True).to(device)
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)