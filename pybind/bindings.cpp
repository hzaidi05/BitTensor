#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/torch.h>
#include <torch/extension.h>
#include <packed_tensor.hpp>
#include <kernels/packed_ops.hpp>

namespace py = pybind11;
using namespace BitTensor;

// Helper function to convert PyTorch tensor to PackedTensor
template<typename T>
PackedTensor<T> from_pytorch_tensor(const torch::Tensor& tensor) {
    return PackedTensor<T>::from_pytorch(tensor);
}

// Helper function to convert PackedTensor to PyTorch tensor
template<typename T>
torch::Tensor to_pytorch_tensor(const PackedTensor<T>& tensor) {
    return tensor.to_pytorch();
}

// Helper function to perform GEMM with PyTorch tensors
torch::Tensor packed_gemm_pytorch(const torch::Tensor& a, const torch::Tensor& b) {
    auto packed_a = PackedTensor<uint8_t>::from_pytorch(a);
    auto packed_b = PackedTensor<uint8_t>::from_pytorch(b);
    PackedTensor<uint8_t> packed_c({a.size(0), b.size(1)});
    
    packed_gemm(packed_a, packed_b, packed_c);
    return packed_c.to_pytorch();
}

PYBIND11_MODULE(bittensor_py, m) {
    m.doc() = "BitTensor: Fast int4 inference library"; 

    // Expose PackedTensor class
    py::class_<PackedTensor<uint8_t>>(m, "PackedTensor")
        .def(py::init<const std::vector<int64_t>&>())
        .def("shape", &PackedTensor<uint8_t>::shape)
        .def("size", &PackedTensor<uint8_t>::size)
        .def("__call__", [](const PackedTensor<uint8_t>& self, const std::vector<int64_t>& indices) {
            return self(indices);
        })
        .def("set", [](PackedTensor<uint8_t>& self, const std::vector<int64_t>& indices, uint8_t value) {
            self.set(indices, value);
        })
        .def_static("from_pytorch", &from_pytorch_tensor<uint8_t>)
        .def("to_pytorch", &to_pytorch_tensor<uint8_t>);

    // Expose GEMM operation
    m.def("packed_gemm", &packed_gemm_pytorch, "Perform GEMM operation with int4 clamping",
          py::arg("a"), py::arg("b"));

    // Add convenience function for direct PyTorch tensor operations
    m.def("matmul", &packed_gemm_pytorch, "Alias for packed_gemm for PyTorch compatibility",
          py::arg("a"), py::arg("b"));
}
