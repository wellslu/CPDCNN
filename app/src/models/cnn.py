import mlconfig
from torch import nn
import torch.nn.functional as F


@mlconfig.register
class CNN(nn.Module):

    def __init__(self, num_filters, filter_h, filter_w, image_channels, rank, num_class):
        super(CNN, self).__init__()

        layers = [nn.Conv2d(image_channels, num_filters, kernel_size=filter_h, bias=False),
                  nn.Conv2d(num_filters, num_filters, kernel_size=filter_h, bias=False)]
        self.features = nn.Sequential(*layers)

        # self.flatten_shape = int(((input_shape[-2]-2)/2-2)/2) * int(((input_shape[-1]-2)/2-2)/2) * num_filters

        self.classifier = nn.Sequential(*[nn.Linear(25088, 50),
                                            nn.Linear(50, num_class)],
                                            nn.LogSoftmax(dim=1))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = nn.AdaptiveAvgPool2d((7,7))
        x = x.view(x.size(0), -1)
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