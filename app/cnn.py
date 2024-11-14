import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V 
import numpy as np
np.random.seed(48)
torch.manual_seed(48)

class CNN(nn.Module):
    # add one more argrument for callback slef-implement conv function
    def __init__(self, num_filters, filter_h, filter_w, image_channels, rank, devices, num_class, input_shape):
        super(CNN, self).__init__()
        # revise those two functions
        self.conv1 = nn.Conv2d(image_channels, num_filters, kernel_size=filter_h, bias=False)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=filter_h, bias=False)
        #self.flatten_shape = int((input_shape[-2]-2)/2) * int((input_shape[-1]-2)/2) * num_filters
        self.flatten_shape = int(((input_shape[-2]-2)/2-2)/2) * int(((input_shape[-1]-2)/2-2)/2) * num_filters
        #self.fc1 = nn.Linear(self.flatten_shape, num_class)
        self.fc1 = nn.Linear(self.flatten_shape, 50)
        self.fc2 = nn.Linear(50, num_class)
        self.num_class = num_class
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1, self.flatten_shape)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x)