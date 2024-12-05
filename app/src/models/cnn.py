import mlconfig
from torch import nn
import torch
import torch.nn.functional as F

class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 初始化卷积核权重，假设使用标准正态分布
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
    def forward(self, x):
        # 获取输入和权重的形状
        batch_size, in_channels, height, width = x.shape
        out_channels, in_channels, kernel_h, kernel_w = self.weight.shape
        
        # 展开输入张量，计算卷积操作
        # 对输入进行 unfold 操作，将卷积区域展开
        x_unfolded = F.unfold(x, kernel_size=(kernel_h, kernel_w))
        
        # 调整展开后的形状以方便后续操作
        x_unfolded = x_unfolded.view(batch_size, in_channels, kernel_h, kernel_w, -1)
        x_unfolded = x_unfolded.permute(0, 4, 1, 2, 3)  # (batch_size, num_windows, in_channels, kernel_h, kernel_w)

        # 使用 torch.einsum 进行卷积操作
        result = torch.einsum('bnchw, ochw -> bon', x_unfolded, self.weight)

        # 将结果 reshape 为输出的形状
        output_h = height - kernel_h + 1
        output_w = width - kernel_w + 1
        result = result.view(batch_size, out_channels, output_h, output_w)
        
        return result


@mlconfig.register
class CNN(nn.Module):

    def __init__(self, num_filters, filter_h, image_channels, num_class):
        super(CNN, self).__init__()

        layers = [
                    nn.Conv2d(image_channels, num_filters, kernel_size=filter_h, bias=False),
                    nn.Conv2d(num_filters, num_filters, kernel_size=filter_h, bias=False)
                    # ConvolutionLayer(image_channels, num_filters, filter_h),
                    # ConvolutionLayer(num_filters, num_filters, filter_h),
                  ]
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(*[nn.Linear(25088, num_class),
                                            # nn.Linear(50, num_class)
                                            ],
                                            nn.LogSoftmax(dim=1))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = nn.AdaptiveAvgPool2d((7,7))
        x = x.reshape(x.size(0), -1)
        # x = x.view(-1, self.flatten_shape)
        x = self.classifier(x)

        return x
    
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)