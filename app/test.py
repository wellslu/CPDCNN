import sys
import os
import torch
import util

start_time = util.get_time()
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

end_time = util.get_time()
time_interval = end_time - start_time
print(f"interval {time_interval}")

matrix = util.Matrix(1024,1,1024,1)
print(f"{matrix.getA()}")