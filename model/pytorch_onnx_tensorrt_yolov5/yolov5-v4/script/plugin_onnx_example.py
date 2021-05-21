import torch
import torch.nn.functional as F
import torch.nn as nn

class SiLUImplementtation(torch.autograd.Function):
    # 主要是这里，对于autograd.Function这种自定义实现的op，只需要添加静态方法symbolic即可，除了g以外的参数应与forward函数的除ctx以外完全一样
    #“SiLU”作为插件名称
    @staticmethod
    def symbolic(g, input):
        return g.op("SiLU", input)

    def forward(self, x):
        return x * torch.sigmoid(x)

    #省略了backward

class customSiLU(nn.Module):
    def forward(self, x):
        return SiLUImplementtation.apply(x)


class FooModel(torch.nn.Module):
    def __init__(self):
        super(FooModel, self).__init__()
        self.SiLU = customSiLU()

    def forward(self, input1, input2):
        return input2 + self.SiLU(input1)


dummy_input1 = torch.zeros((1, 3, 3, 3))
dummy_input2 = torch.zeros((1, 1, 3, 3))
model = FooModel()

# 这里演示了2个输入的情况，实际上你可以自己定义几个输入
# torch高版本需添加operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK来导出自定义层，参见torch.onnx官方文档
torch.onnx.export(model, (dummy_input1, dummy_input2), 'test.onnx', verbose=True, opset_version=12,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)