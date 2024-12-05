import torch
import time
from src.models.cnn import CNN
from src.models.cnn_decomp import CNN_decomp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.ones((32, 1, 32, 32)).float().to(device)

model = CNN_decomp(32, 3, 3, 1, 2, 10)
# model.load_state_dict(torch.load("./model/best_cnnDecomp_2.pth", weights_only=True))
model = model.to(device)
model.eval()

start = time.time()
for i in range(1):
    model(data)
end = time.time()
print(end-start)
with open("./cnnDecomp2.txt", "a") as f:
    f.write(str(end-start))
    f.write("\t")