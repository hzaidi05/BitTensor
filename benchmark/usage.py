import torch
import torch.nn as nn
import torch.quantization
import time
import bittensor

class PackedLinear(nn.Module):
    def __init__(self, weight_tensor: torch.Tensor, bias=None):
        super().__init__()
        # Ensure weights are int4 (0-15)
        if not torch.all((weight_tensor >= 0) & (weight_tensor <= 15)):
            raise ValueError("Weights must be in range [0, 15]")
        
        # Pack weights once during initialization
        self.weight_packed = bittensor.pack_weights(weight_tensor)
        self.bias = bias
        self.in_features = weight_tensor.size(1)
        self.out_features = weight_tensor.size(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is int4
        if not torch.all((x >= 0) & (x <= 15)):
            raise ValueError("Input must be in range [0, 15]")
        
        packed_out = bittensor.packed_matmul(x, self.weight_packed, self.bias)
        return bittensor.unpack_tensor(packed_out, x.size(0), self.out_features)

# Create quantized int8 Linear using PyTorch
def create_quantized_linear(in_features, out_features, weight_fp32, bias_fp32):
    float_linear = nn.Linear(in_features, out_features)
    float_linear.weight.data = weight_fp32.clone()
    float_linear.bias.data = bias_fp32.clone()

    float_linear.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(float_linear, inplace=True)

    dummy_input = torch.randn(1, in_features)
    float_linear(dummy_input)

    quantized_linear = torch.quantization.convert(float_linear, inplace=False)
    return quantized_linear

def profile_linear_layers():
    print("\nProfiling Linear Layers (Packed int4 vs Quantized int8):")
    print("-" * 60)

    batch_size = 32
    in_features = 512
    out_features = 512
    num_runs = 100

    # Generate random float32 data
    x_fp32 = torch.randn(batch_size, in_features)
    w_fp32 = torch.randn(out_features, in_features)
    b_fp32 = torch.randn(out_features)

    # Simulate quantization for PackedLinear (int4)
    scale = 1.0  # pretend scale for now
    x_q = (x_fp32 / scale).round().clamp(0, 15).to(torch.uint8)
    w_q = (w_fp32 / scale).round().clamp(0, 15).to(torch.uint8)
    b_q = (b_fp32 / scale).round().clamp(0, 15).to(torch.uint8)

    # Create layers
    packed_layer = PackedLinear(w_q, b_q)
    quant_layer = create_quantized_linear(in_features, out_features, w_fp32, b_fp32)

    # Warmup
    for _ in range(10):
        _ = packed_layer(x_q)
        _ = quant_layer(x_fp32)

    def benchmark(model, x, name):
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = model(x)
            times.append((time.time() - start) * 1000)
        print(f"{name:<20}: {sum(times)/len(times):.2f} ms")

    print(f"Input shape: {x_fp32.shape}")
    print(f"Weight shape: {w_fp32.shape}")
    print(f"Parameters: {w_fp32.numel():,}")

    print(f"Packed weight shape: {packed_layer.weight_packed.shape}")
    print(f"Memory reduction: {w_q.numel() / packed_layer.weight_packed.numel():.1f}x")

    print("\nAverage inference time (ms):")
    benchmark(packed_layer, x_q, "PackedLinear (int4)")
    benchmark(quant_layer, x_fp32, "QuantizedLinear (int8)")

    # Output similarity check (optional)
    out_int4 = packed_layer(x_q)
    out_int8 = quant_layer(x_fp32)
    diff = torch.mean((out_int4.float() - out_int8.float()).abs()).item()
    print(f"\nOutput L1 diff (mean absolute error): {diff:.4f}")

def main():
    in_features = 16
    out_features = 8
    weight = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
    bias = torch.randint(0, 16, (out_features,), dtype=torch.uint8)
    x = torch.randint(0, 16, (1, in_features), dtype=torch.uint8)

    layer = PackedLinear(weight, bias)
    out = layer(x)

    print("==== PackedLinear Static Example ====")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Original weight shape: {weight.shape}")
    print(f"Packed weight shape: {layer.weight_packed.shape}")
    print(f"Memory reduction: {weight.numel() / layer.weight_packed.numel():.1f}x")

    profile_linear_layers()

if __name__ == "__main__":
    main()
