import torch
import sys
sys.path.append("/mnt/c/Users/cooki/OneDrive/Desktop/Projects/BitTensor/build")
import bittensor_py as bt

def main():
    # Create some test tensors (values in int4 range 0-15)
    a = torch.tensor([
        [1, 2, 3],
        [2, 3, 4]
    ], dtype=torch.int64)
    
    b = torch.tensor([
        [2, 3],
        [3, 4],
        [4, 5]
    ], dtype=torch.int64)
    
    # Method 1: Direct PyTorch-like usage
    c1 = bt.matmul(a, b)
    print("Direct matmul result:")
    print(c1)
    
    # Method 2: Using PackedTensor class
    packed_a = bt.PackedTensor.from_pytorch(a)
    packed_b = bt.PackedTensor.from_pytorch(b)
    packed_c = bt.PackedTensor([2, 2])
    bt.packed_gemm(packed_a, packed_b, packed_c)
    c2 = packed_c.to_pytorch()
    print("\nPackedTensor result:")
    print(c2)
    
    # Compare with PyTorch's result (without clamping)
    torch_c = torch.matmul(a, b)
    print("\nPyTorch result (without clamping):")
    print(torch_c)
    
    # Verify all values are in int4 range
    print("\nVerifying int4 range (0-15):")
    print("Direct matmul max value:", c1.max().item())
    print("PackedTensor max value:", c2.max().item())
    print("PyTorch max value:", torch_c.max().item())

if __name__ == "__main__":
    main() 