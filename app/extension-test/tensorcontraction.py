import torch
import util

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
input_tensor = torch.arange(1, 2 * 2 * 2 * 2 * 3 * 2 + 1).reshape(2, 2, 2, 2, 3, 2).float()

# Factors
factors = [
    create_diagonal_matrix(3, 2),
    create_diagonal_matrix(2, 2),
    create_diagonal_matrix(3, 2),
    create_diagonal_matrix(2, 2),
]

# Output initialization
output = None

# Compute einsum and sum over batches
for i in range(2):  # Iterate over columns of the last factor
    # First contraction
    result = torch.einsum('abcdef,f->abcde', input_tensor, factors[3][:, i])
    # Second contraction
    result = torch.einsum('abcde,e->abcd', result, factors[2][:, i])
    # Third contraction
    result = torch.einsum('abcd,d->abc', result, factors[1][:, i])
    # Final contraction
    result = torch.einsum('abc,t->abct', result, factors[0][:, i])
    
    # Aggregate results
    if output is None:
        output = result
    else:
        output += result

# Print the output
print("Final Output:")
print(output)

Cuda_Output = util.tensorcontraction(input_tensor, factors)
Cuda_Output = Cuda_Output.transpose(0, 1).contiguous()
print("Cuda based Output:")
print(Cuda_Output)
