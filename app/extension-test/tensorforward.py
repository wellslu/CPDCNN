import torch
from util import Cuutil
import time
import argparse

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
input_tensor = torch.randn(16, 1, 30, 30).float() # 4-way tensor (Batch, Channel, H, W)

def main():
    # Factors
    factors = [
        create_diagonal_matrix(32, 2), # Output Channel
        create_diagonal_matrix(3, 2),  # H filter
        create_diagonal_matrix(3, 2),  # W filter
        create_diagonal_matrix(1, 2),  # Input Channel
    ]
    # Move each tensor in the list to the GPU
    factors = [factor.cuda() for factor in factors]
    # Output initialization
    output = None
    # round = 1 #1e5    
    
    #Cuda Part
    if independent != "Y":
        current = input_tensor

    original_start = time.time()
    for _ in range(round):

        # reshpae input tensor
        current = torch.nn.functional.pad(input_tensor if independent == "Y" else current, (1, 1, 1, 1)) # if truly for next round should pad current
        current = current.unfold(2, size=factors[1].size(0), step=1).unfold(3, size=factors[2].size(0), step=1)
        current = current.permute(0,2,3,4,5,1).cuda() # 6-way (Batch, H_new, W_new, H, W, Channel)
        
        # Compute einsum and sum over batches
        inner_current = None
        for i in range(2):  # Iterate over columns of the last factor
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

        # reshape back from 4-way (Batch, H, W, Channnel) to 4-way (Batch, Channel, H, W) for the next-round
        current = current.permute(0,3,1,2)
        
    output = current    
    original_end = time.time()

    # Print the output
    print(f"******Round [{round}], Independent [{independent}] ********")
    print("Orignal Final Output:")
    print(output.shape)
    print(original_end-original_start)
    # print(output)


    #Cuda Part
    if independent != "Y":
        current = input_tensor

    cuutil = Cuutil()
    original_start = time.time()
    for _ in range(round):

        # reshpae input tensor if independent == "Y" else current
        current = torch.nn.functional.pad(input_tensor if independent == "Y" else current, (1, 1, 1, 1)) # if truly for next round should pad current
        current = current.unfold(2, size=factors[1].size(0), step=1).unfold(3, size=factors[2].size(0), step=1)
        current = current.permute(0,2,3,4,5,1).cuda() # 6-way (Batch, H_new, W_new, H, W, Channel)
        
        current = cuutil.tensorcontraction(current, factors)
        current = current.transpose(0, 1).contiguous()
        current = current.reshape(input_tensor.size(0), input_tensor.size(2), input_tensor.size(3), factors[0].size(0))
        
        # reshape back from 4-way (Batch, H, W, Channnel) to 4-way (Batch, Channel, H, W) for the next-round
        current = current.permute(0,3,1,2)
        
    output = current    
    original_end = time.time()

    # Print the output
    print("Cuda-based Final Output:")
    print(output.shape)
    print(original_end-original_start)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Perform einsum v.s. cuda-optimized')

    # Add arguments
    parser.add_argument('-i', '--independent', type=str, help='recursive using the current tensor or retieve the same input tensor')
    parser.add_argument('-r', '--round', type=int)

    args = parser.parse_args()
    round = int(args.round)
    independent = args.independent
    main()