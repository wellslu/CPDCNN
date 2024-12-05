import torch
import util
import time

def create_diagonal_matrix(rows, cols):
    """
    Create a diagonal matrix with specified rows and columns.
    If rows > cols, pads with zeros at the bottom.
    If cols > rows, pads with zeros to the right.
    """
    diag = torch.eye(min(rows, cols)).float()
    if rows > cols:
        return torch.cat([diag, torch.zeros(rows - cols, cols).float()], dim=0)
    elif cols > rows:
        return torch.cat([diag, torch.zeros(rows, cols - rows).float()], dim=1)
    return diag

# Input tensor
# input_tensor = torch.arange(1, 2 * 2 * 2 * 2 * 3 * 2 + 1).reshape(2, 2, 2, 2, 3, 2).float()
input_tensor = torch.randn(16, 30, 30, 3, 3, 1).float()

# Factors
factors = [
    create_diagonal_matrix(32, 2),
    create_diagonal_matrix(3, 2),
    create_diagonal_matrix(3, 2),
    create_diagonal_matrix(1, 2),
]


    

# Output initialization
output = None

current = input_tensor

original_start = time.time()
for _ in range(1):
    current = current.unfold(1, size=factors[1].size(0), step=1).unfold(2, size=factors[2].size(0), step=1)
    # Compute einsum and sum over batches
    inner_current = None
    for i in range(3):  # Iterate over columns of the last factor
        result = torch.einsum('Babcde,e->Babcd', current, factors[3][:, i])
        result = torch.einsum('Babcd,d->Babc', result, factors[2][:, i])
        result = torch.einsum('Babc,c->Bab', result, factors[1][:, i])
        result = torch.einsum('Bab,c->Babc', result, factors[0][:, i])
        
        # Aggregate results
        if inner_current is None:
            inner_current = result
        else:
            inner_current += result
    
    current = inner_current

    
output = current    
original_end = time.time()

# Print the output
print("Final Output:")
print(output.shape)
print(original_end-original_start)
# print(output)

# cuda_start = time.time()
# for _ in range(1):
#     Cuda_Output = util.tensorcontraction(input_tensor, factors)
# cuda_end = time.time()
# Cuda_Output = Cuda_Output.transpose(0, 1).contiguous()
# Cuda_Output = Cuda_Output.reshape(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), factors[0].size(0))
# print("Cuda based Output:")
# print(Cuda_Output.shape)
# print(cuda_end-cuda_start)
