# BitTensor

A high-performance C++ library for bit-packed tensor operations, with PyTorch integration for efficient int4 inference on CPUs.

Modern deep learning models are increasingly being quantized to reduce memory usage and improve inference speed on consumer/edge devices. While GPUs have excellent support for int8 operations, CPUs often lack efficient int4 support. BitTensor addresses this by:

1. Packing 16 int4 values into a single int64 word, enabling vectorized loads
2. Using template metaprogramming for compile-time optimizations
3. Providing a seamless PyTorch integration through C++ extensions

This approach allows for efficient int4 inference on CPUs, with potential memory savings of up to 16x compared to float32.

## Basic Usage

```python
import torch
import bittensor

# Create a quantized linear layer
class PackedLinear(torch.nn.Module):
    def __init__(self, weight_tensor: torch.Tensor, bias=None):
        super().__init__()
        # Pack weights once during initialization
        self.weight_packed = bittensor.pack_weights(weight_tensor)
        self.bias = bias
        self.in_features = weight_tensor.size(1)
        self.out_features = weight_tensor.size(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        # Perform packed matrix multiplication
        packed_out = bittensor.packed_matmul(x, self.weight_packed, self.bias)
        #Unpack output
        return bittensor.unpack_tensor(packed_out, x.size(0), self.out_features)

# Example usage
weight = torch.randint(0, 16, (8, 16), dtype=torch.uint8)  # int4 weights
bias = torch.randint(0, 16, (8,), dtype=torch.uint8)      # int4 bias
layer = PackedLinear(weight, bias)

# Forward pass with int4 input
x = torch.randint(0, 16, (1, 16), dtype=torch.uint8)
output = layer(x)
```

## Roadmap

- [ ] Enable autograd for training support
- [ ] Add `torch.compile` integration for it to go brr
- [ ] Add int4 inference example with MNIST
- [ ] SIMD optimizations
- [ ] More operations and fused kernels

Performance results coming soon!