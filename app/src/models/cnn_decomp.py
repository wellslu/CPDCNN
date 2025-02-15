import mlconfig
import torch
from torch import nn
import torch.nn.functional as F
from tensorly.decomposition import parafac
import tensorly as tl
import numpy as np
import util

class DecompLayer(nn.Module):
    def __init__(self, num_filters, filter_h, filter_w, original_channels, rank):
        super(DecompLayer, self).__init__()
        filters = torch.randn(num_filters, filter_h, filter_w, original_channels) / (filter_h*filter_w)
        tensor = tl.tensor(filters)
        
        _, factors = parafac(tensor, rank)

        self.factors = nn.ParameterList([
            nn.Parameter(torch.tensor(factors[0], dtype=torch.float32)),
            nn.Parameter(torch.tensor(factors[1], dtype=torch.float32)),
            nn.Parameter(torch.tensor(factors[2], dtype=torch.float32)),
            nn.Parameter(torch.tensor(factors[3], dtype=torch.float32))
        ])
        self.num_filters = num_filters
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.rank = rank

    def forward(self, x):
        # input_shape = x.shape #B,C,H,W
        x = x.unfold(2, self.filter_h, 1).unfold(3, self.filter_w, 1)
        shape = x.shape

        # x = x.reshape(shape[0], shape[1], shape[2]*shape[3], shape[4],shape[5])#B,C,N,h,w
        # x = x.permute(0, 2, 3, 4, 1)#B,N,h,w,C
        # output = torch.tensor(np.zeros((shape[0], (shape[2])*(shape[3]), self.num_filters)), dtype=x.dtype).to("cuda")
        # # self.output = torch.tensor(np.zeros((shape[0], (shape[2])*(shape[3]), self.num_filters)), dtype=x.dtype)

        # for i in range(self.rank):
        #     result = torch.einsum('Babcd,Bd->Babc', x.float(), self.factors[3][:,i].repeat(shape[0], 1))
        #     result = torch.einsum('Babc,Bc->Bab', result, self.factors[2][:,i].repeat(shape[0], 1))
        #     result = torch.einsum('Bab,Bb->Ba', result, self.factors[1][:,i].repeat(shape[0], 1))
        #     result = torch.einsum('Ba,Bb->Bab', result, self.factors[0][:,i].repeat(shape[0], 1))

        #     output += result

        x = x.permute(0, 2, 3, 4, 5, 1).contiguous() #B,H,W,h,w,C
        output = util.tensorcontraction(x, self.factors).to("cuda")
        output = output.transpose(0, 1)
        
        output = output.reshape((shape[0], shape[2], shape[3], self.num_filters))
        output = output.permute(0,3,1,2)
        return output


@mlconfig.register
class CNN_decomp(nn.Module):

    def __init__(self, num_filters, filter_h, filter_w, image_channels, rank, num_class):
        super(CNN_decomp, self).__init__()

        self.decomp_layer1 = DecompLayer(num_filters, filter_h, filter_w, image_channels, rank)
        self.decomp_layer2 = DecompLayer(num_filters, filter_h, filter_w, num_filters, rank)

        self.classifier = nn.Sequential(nn.Linear(25088, num_class),
                                        nn.LogSoftmax(dim=1))
        self._initialize_weights()

    def forward(self, x):
        x = self.decomp_layer1(x)
        x = self.decomp_layer2(x)

        x = x.reshape(x.size(0), -1)

        x = self.classifier(x)

        return x
    
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)