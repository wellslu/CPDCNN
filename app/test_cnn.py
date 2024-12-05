import torch
import time
from src.models.cnn import CNN
from src.models.cnn_decomp import CNN_decomp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.randn(32, 1, 32, 32).float().to(device)

model = torch.load("./model/best_cnn.pt")
model = model.to(device)
model.eval()

start = time.time()
for i in range(100):
    model(data)
end = time.time()
print(end-start)
with open("./cnn.txt", "a") as f:
    f.write(str(end-start))
    f.write("\t")