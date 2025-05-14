#include "packed_tensor.hpp"
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace BitTensor {
namespace torch {

at::Tensor pack_weights(const at::Tensor& weight) {
    auto weight_cpu = weight.contiguous().cpu();
    
    if (weight_cpu.scalar_type() != at::ScalarType::Byte) {
        throw std::runtime_error("Weight must be quantized to uint8 (int4) before packing");
    }
    
    if (weight_cpu.dim() != 2) {
        throw std::runtime_error("Weight must be 2D tensor [out_features, in_features]");
    }
    
    const int64_t out_features = weight_cpu.size(0);
    const int64_t in_features = weight_cpu.size(1);
    
    int64_t num_words = (in_features + 15) / 16;
    PackedTensor<uint8_t> packed({out_features, num_words});

    
    auto* data = weight_cpu.data_ptr<uint8_t>();
    
    // For each output feature
    for (int64_t i = 0; i < out_features; ++i) {
        // Pack 16 input features at a time
        const int64_t num_words = (in_features + 15) / 16;
        for (int64_t j = 0; j < num_words; ++j) {
            uint8_t values[16] = {0};
            const int64_t start_idx = j * 16;
            const int64_t end_idx = std::min(start_idx + 16, in_features);
            
            for (int64_t k = start_idx; k < end_idx; ++k) {
                values[k - start_idx] = data[i * in_features + k] & 0xF;
            }
            
            packed.data()[i * num_words + j] = detail::pack_16_values_impl(values);
        }
    }
    
    // Create output tensor with packed data
    auto options = at::TensorOptions().dtype(at::ScalarType::Long).device(at::kCPU);

    std::vector<int64_t> packed_shape = {out_features, (in_features + 15) / 16};
    auto packed_tensor = at::empty(packed_shape, options);
    auto* packed_data = packed_tensor.data_ptr<uint64_t>();
    
    std::memcpy(packed_data, packed.data(), packed.data_size() * sizeof(uint64_t));
    
    return packed_tensor;
}

at::Tensor unpack_weights(const at::Tensor& packed_weights, int64_t in_features) {
    auto packed_cpu = packed_weights.contiguous().cpu();
    
    if (packed_cpu.dim() != 2) {
        throw std::runtime_error("Packed weights must be 2D tensor [out_features, in_features/16]");
    }
    
    const int64_t out_features = packed_cpu.size(0);
    const int64_t num_words = packed_cpu.size(1);
    
    if (num_words != (in_features + 15) / 16) {
        throw std::runtime_error("Packed weights dimensions do not match in_features");
    }
    
    auto options = at::TensorOptions()
        .dtype(at::ScalarType::Byte)
        .device(at::kCPU);
    
    auto output = at::empty({out_features, in_features}, options);
    auto* output_data = output.data_ptr<uint8_t>();
    
    auto* packed_data = packed_cpu.data_ptr<uint64_t>();
    
    for (int64_t i = 0; i < out_features; ++i) {
        for (int64_t j = 0; j < num_words; ++j) {
            uint8_t values[16];
            detail::unpack_16_values_impl(packed_data[i * num_words + j], values);
            
            const int64_t start_idx = j * 16;
            const int64_t end_idx = std::min(start_idx + 16, in_features);
            
            for (int64_t k = start_idx; k < end_idx; ++k) {
                output_data[i * in_features + k] = values[k - start_idx];
            }
        }
    }
    
    return output;
}


at::Tensor packed_matmul(const at::Tensor& input, const at::Tensor& packed_weights, const at::Tensor& bias) {
    // Ensure input is contiguous and on CPU
    auto input_cpu = input.contiguous().cpu();
    
    // Verify input dimensions and type
    if (input_cpu.dim() != 2) {
        throw std::runtime_error("Input must be 2D tensor [batch_size, in_features]");
    }
    if (input_cpu.scalar_type() != at::ScalarType::Byte) {
        throw std::runtime_error("Input must be quantized to uint8 (int4)");
    }
    
    const int64_t batch_size = input_cpu.size(0);
    const int64_t in_features = input_cpu.size(1);
    const int64_t out_features = packed_weights.size(0);
    
    // Verify packed weights dimensions
    if (packed_weights.size(1) != (in_features + 15) / 16) {
        throw std::runtime_error("Packed weights dimensions do not match input features");
    }
    
    PackedTensor<uint8_t> packed_input({batch_size, in_features});
    auto* input_data = input_cpu.data_ptr<uint8_t>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t num_words = (in_features + 15) / 16;
        for (int64_t j = 0; j < num_words; ++j) {
            uint8_t values[16] = {0};
            const int64_t start_idx = j * 16;
            const int64_t end_idx = std::min(start_idx + 16, in_features);
            
            for (int64_t k = start_idx; k < end_idx; ++k) {
                values[k - start_idx] = input_data[b * in_features + k] & 0xF;
            }
            
            packed_input.data()[b * num_words + j] = detail::pack_16_values_impl(values);
        }
    }
    
    int64_t num_words = (in_features + 15) / 16;
    PackedTensor<uint8_t> packed_w({out_features, num_words});

    auto* packed_w_data = packed_weights.data_ptr<uint64_t>();
    std::memcpy(packed_w.data(), packed_w_data, packed_weights.numel() * sizeof(uint64_t));
    
    auto options = at::TensorOptions()
        .dtype(at::ScalarType::Long)
        .device(at::kCPU);
    
    auto output = at::empty({batch_size, out_features}, options);
    auto* output_data = output.data_ptr<uint64_t>();
    
    PackedTensor<uint8_t> packed_output({batch_size, out_features});
    
    uint8_t* bias_data = nullptr;
    if (bias.numel() > 0) {
        auto bias_cpu = bias.contiguous().cpu();
        bias_data = bias_cpu.data_ptr<uint8_t>();
    }
    
    BitTensor::packed_gemm(packed_input, packed_w, packed_output, bias_data);
    
    std::memcpy(output_data, packed_output.data(), packed_output.data_size() * sizeof(uint64_t));
    
    return output;
}

// Register operators with PyTorch
TORCH_LIBRARY(bittensor, m) {
    m.def("pack_weights(Tensor weight) -> Tensor", &pack_weights);
    m.def("unpack_weights(Tensor packed_weights, int in_features) -> Tensor", &unpack_weights);
    m.def("packed_matmul(Tensor input, Tensor packed_weights, Tensor? bias) -> Tensor", &packed_matmul);
}

} // namespace torch
} // namespace BitTensor

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}
