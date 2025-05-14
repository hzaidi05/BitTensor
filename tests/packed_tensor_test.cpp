#include <gtest/gtest.h>
#include <packed_tensor.hpp>
#include <kernels/packed_ops.hpp>
#include <torch/torch.h>

using namespace BitTensor;

TEST(PackedTensorTest, BasicOperations) {
    // Test dynamic tensor
    std::vector<int64_t> shape = {2, 3, 4};
    PackedTensor<uint8_t> tensor(shape);
    
    // Test shape
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 24); // 2 * 3 * 4
    
    // Test element access
    std::vector<int64_t> indices = {1, 2, 3};
    tensor.set(indices, 5);
    EXPECT_EQ(tensor(indices), 5);
    
    // Test linear access
    tensor.set_linear(0, 1);
    EXPECT_EQ(tensor.get_linear(0), 1);
}

TEST(PackedTensorTest, PyTorchConversion) {
    // Create a PyTorch tensor
    auto torch_tensor = torch::tensor({
        {1, 2, 3},
        {4, 5, 6}
    }, torch::kInt64);
    
    // Convert to packed tensor
    auto packed = PackedTensor<uint8_t>::from_pytorch(torch_tensor);
    
    // Check shape
    EXPECT_EQ(packed.shape().size(), 2);
    EXPECT_EQ(packed.shape()[0], 2);
    EXPECT_EQ(packed.shape()[1], 3);
    
    // Check values
    EXPECT_EQ(packed({0, 0}), 1);
    EXPECT_EQ(packed({0, 1}), 2);
    EXPECT_EQ(packed({1, 2}), 6);
    
    // Convert back to PyTorch
    auto converted_back = packed.to_pytorch();
    
    // Check equality
    EXPECT_TRUE(torch::equal(torch_tensor, converted_back));
}

TEST(PackedTensorStaticTest, BasicOperations) {
    PackedTensorStatic<2, 3, 4> tensor;
    const auto& shape = tensor.get_shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(tensor.get_size(), 24);
    EXPECT_EQ(tensor.get_numel(), 2); // 24 elements / 16 elements per word = 2 words

    // Test setting and getting values
    std::array<int64_t, 3> indices = {0, 0, 0};
    tensor.set(indices, 1);
    EXPECT_EQ(tensor(indices), 1);

    indices = {1, 2, 3};
    tensor.set(indices, 15);
    EXPECT_EQ(tensor(indices), 15);
}

TEST(PackedTensorStaticTest, PyTorchConversion) {
    // Create a PyTorch tensor
    auto torch_tensor = torch::tensor({
        {1, 2, 3},
        {4, 5, 6}
    }, torch::kInt64);
    
    // Convert to packed tensor
    auto packed = PackedTensorStatic<2, 3>::from_pytorch(torch_tensor);
    
    // Check values
    EXPECT_EQ(packed({0, 0}), 1);
    EXPECT_EQ(packed({0, 1}), 2);
    EXPECT_EQ(packed({1, 2}), 6);
    
    // Convert back to PyTorch
    auto converted_back = packed.to_pytorch();
    
    // Check equality
    EXPECT_TRUE(torch::equal(torch_tensor, converted_back));
}

TEST(PackedTensorTest, OutOfBoundsAccess) {
    // Test dynamic tensor
    std::vector<int64_t> shape = {2, 3, 4};
    PackedTensor<uint8_t> tensor(shape);
    
    // Test out of bounds access
    EXPECT_THROW(tensor({2, 3, 4}), std::out_of_range);
    EXPECT_THROW(tensor({-1, 0, 0}), std::out_of_range);
    EXPECT_THROW(tensor({0, 3, 0}), std::out_of_range);
    EXPECT_THROW(tensor({0, 0, 4}), std::out_of_range);
    
    // Test out of bounds set
    EXPECT_THROW(tensor.set({2, 3, 4}, 0), std::out_of_range);
    EXPECT_THROW(tensor.set({-1, 0, 0}, 0), std::out_of_range);
    
    // Test linear access out of bounds
    EXPECT_THROW(tensor.get_linear(24), std::out_of_range);
    EXPECT_THROW(tensor.set_linear(24, 0), std::out_of_range);
}

TEST(PackedTensorTest, BitPackingAccuracy) {
    // Test dynamic tensor
    std::vector<int64_t> shape = {2, 3, 4};
    PackedTensor<uint8_t> tensor(shape);
    
    // Test basic 4-bit values
    tensor.set_linear(0, 15);  // 0b1111
    tensor.set_linear(1, 1);   // 0b0001
    EXPECT_EQ(tensor.get_linear(0), 15);
    EXPECT_EQ(tensor.get_linear(1), 1);
    
    // Test overlapping values don't interfere
    tensor.set_linear(0, 15);
    tensor.set_linear(1, 0);
    EXPECT_EQ(tensor.get_linear(0), 15);
    EXPECT_EQ(tensor.get_linear(1), 0);
    
    // Test all possible 4-bit values
    for (uint8_t i = 0; i < 16; ++i) {
        tensor.set_linear(0, i);
        EXPECT_EQ(tensor.get_linear(0), i);
    }
    
    // Test adjacent values in same word
    tensor.set_linear(0, 15);
    tensor.set_linear(1, 0);
    tensor.set_linear(2, 8);
    EXPECT_EQ(tensor.get_linear(0), 15);
    EXPECT_EQ(tensor.get_linear(1), 0);
    EXPECT_EQ(tensor.get_linear(2), 8);
}

TEST(PackedTensorStaticTest, OutOfBoundsAccess) {
    // Test static tensor
    PackedTensorStatic<2, 3, 4> tensor;
    
    // Test out of bounds access
    EXPECT_THROW(tensor({2, 3, 4}), std::out_of_range);
    EXPECT_THROW(tensor({-1, 0, 0}), std::out_of_range);
    EXPECT_THROW(tensor({0, 3, 0}), std::out_of_range);
    EXPECT_THROW(tensor({0, 0, 4}), std::out_of_range);
    
    // Test out of bounds set
    EXPECT_THROW(tensor.set({2, 3, 4}, 0), std::out_of_range);
    EXPECT_THROW(tensor.set({-1, 0, 0}, 0), std::out_of_range);
    
    // Test linear access out of bounds
    EXPECT_THROW(tensor.get_linear(24), std::out_of_range);
    EXPECT_THROW(tensor.set_linear(24, 0), std::out_of_range);
}

TEST(PackedTensorStaticTest, BitPackingAccuracy) {
    // Test static tensor
    PackedTensorStatic<2, 3, 4> tensor;
    
    // Test basic 4-bit values
    tensor.set_linear(0, 15);  // 0b1111
    tensor.set_linear(1, 1);   // 0b0001
    EXPECT_EQ(tensor.get_linear(0), 15);
    EXPECT_EQ(tensor.get_linear(1), 1);
    
    // Test overlapping values don't interfere
    tensor.set_linear(0, 15);
    tensor.set_linear(1, 0);
    EXPECT_EQ(tensor.get_linear(0), 15);
    EXPECT_EQ(tensor.get_linear(1), 0);
    
    // Test all possible 4-bit values
    for (uint8_t i = 0; i < 16; ++i) {
        tensor.set_linear(0, i);
        EXPECT_EQ(tensor.get_linear(0), i);
    }
    
    // Test adjacent values in same word
    tensor.set_linear(0, 15);
    tensor.set_linear(1, 0);
    tensor.set_linear(2, 8);
    EXPECT_EQ(tensor.get_linear(0), 15);
    EXPECT_EQ(tensor.get_linear(1), 0);
    EXPECT_EQ(tensor.get_linear(2), 8);
}

TEST(PackedTensorTest, GEMMBasic) {
    // Test dynamic GEMM with small matrices
    std::vector<int64_t> A_shape = {2, 3};
    std::vector<int64_t> B_shape = {3, 2};
    std::vector<int64_t> C_shape = {2, 2};
    
    PackedTensor<uint8_t> A(A_shape);
    PackedTensor<uint8_t> B(B_shape);
    PackedTensor<uint8_t> C(C_shape);
    
    // Fill A and B with test values (using smaller values to stay in 4-bit range)
    A.set({0, 0}, 1); A.set({0, 1}, 2); A.set({0, 2}, 3);
    A.set({1, 0}, 2); A.set({1, 1}, 3); A.set({1, 2}, 4);
    
    B.set({0, 0}, 2); B.set({0, 1}, 3);
    B.set({1, 0}, 3); B.set({1, 1}, 4);
    B.set({2, 0}, 4); B.set({2, 1}, 5);
    
    // Perform GEMM
    packed_gemm(A, B, C);
    
    // Verify results (all values should be clamped to 4-bit range 0-15)
    EXPECT_EQ(C({0, 0}), 15);  // 1*2 + 2*3 + 3*4 = 20 -> clamped to 15
    EXPECT_EQ(C({0, 1}), 15);  // 1*3 + 2*4 + 3*5 = 26 -> clamped to 15
    EXPECT_EQ(C({1, 0}), 15);  // 2*2 + 3*3 + 4*4 = 29 -> clamped to 15
    EXPECT_EQ(C({1, 1}), 15);  // 2*3 + 3*4 + 4*5 = 38 -> clamped to 15
}

TEST(PackedTensorTest, GEMMPyTorchComparison) {
    // Create PyTorch tensors with smaller values (simulating int4 inputs)
    auto torch_A = torch::tensor({
        {1, 2, 3},
        {2, 3, 4}
    }, torch::kInt64);
    
    auto torch_B = torch::tensor({
        {2, 3},
        {3, 4},
        {4, 5}
    }, torch::kInt64);
    
    // Convert to packed tensors
    auto packed_A = PackedTensor<uint8_t>::from_pytorch(torch_A);
    auto packed_B = PackedTensor<uint8_t>::from_pytorch(torch_B);
    PackedTensor<uint8_t> packed_C({2, 2});
    
    // Perform GEMM with our packed implementation (automatically clamps to int4)
    packed_gemm(packed_A, packed_B, packed_C);
    
    // Simulate int4 inference in PyTorch
    auto torch_C = torch::matmul(torch_A, torch_B);
    // Clamp the result to int4 range (0-15)
    torch_C = torch::clamp(torch_C, 0, 15);
    auto packed_C_torch = packed_C.to_pytorch();
    
    // Both results should now be in int4 range
    EXPECT_TRUE(torch::all(torch_C <= 15).item<bool>());
    EXPECT_TRUE(torch::all(packed_C_torch <= 15).item<bool>());
    
    // And they should match since both are properly clamped
    EXPECT_TRUE(torch::equal(torch_C, packed_C_torch));
}

TEST(PackedTensorTest, GEMMEdgeCases) {
    // Test with zero matrices
    std::vector<int64_t> shape = {2, 2};
    PackedTensor<uint8_t> A(shape);
    PackedTensor<uint8_t> B(shape);
    PackedTensor<uint8_t> C(shape);
    
    packed_gemm(A, B, C);
    EXPECT_EQ(C({0, 0}), 0);
    EXPECT_EQ(C({0, 1}), 0);
    EXPECT_EQ(C({1, 0}), 0);
    EXPECT_EQ(C({1, 1}), 0);
    
    // Test with identity matrix
    A.set({0, 0}, 1); A.set({1, 1}, 1);
    B.set({0, 0}, 1); B.set({1, 1}, 1);
    
    packed_gemm(A, B, C);
    EXPECT_EQ(C({0, 0}), 1);
    EXPECT_EQ(C({0, 1}), 0);
    EXPECT_EQ(C({1, 0}), 0);
    EXPECT_EQ(C({1, 1}), 1);
}

TEST(PackedTensorStaticTest, GEMMBasic) {
    // Test static GEMM with small matrices
    PackedTensorStatic<2, 3> A;
    PackedTensorStatic<3, 2> B;
    PackedTensorStatic<2, 2> C;
    
    // Fill A and B with test values (using smaller values to stay in 4-bit range)
    A.set({0, 0}, 1); A.set({0, 1}, 2); A.set({0, 2}, 3);
    A.set({1, 0}, 2); A.set({1, 1}, 3); A.set({1, 2}, 4);
    
    B.set({0, 0}, 2); B.set({0, 1}, 3);
    B.set({1, 0}, 3); B.set({1, 1}, 4);
    B.set({2, 0}, 4); B.set({2, 1}, 5);
    
    // Perform GEMM
    packed_gemm_static(A, B, C);
    
    // Verify results (all values should be clamped to 4-bit range 0-15)
    EXPECT_EQ(C({0, 0}), 15);  // 1*2 + 2*3 + 3*4 = 20 -> clamped to 15
    EXPECT_EQ(C({0, 1}), 15);  // 1*3 + 2*4 + 3*5 = 26 -> clamped to 15
    EXPECT_EQ(C({1, 0}), 15);  // 2*2 + 3*3 + 4*4 = 29 -> clamped to 15
    EXPECT_EQ(C({1, 1}), 15);  // 2*3 + 3*4 + 4*5 = 38 -> clamped to 15
}

TEST(PackedTensorStaticTest, GEMMPyTorchComparison) {
    // Create PyTorch tensors with smaller values (simulating int4 inputs)
    auto torch_A = torch::tensor({
        {1, 2, 3},
        {2, 3, 4}
    }, torch::kInt64);
    
    auto torch_B = torch::tensor({
        {2, 3},
        {3, 4},
        {4, 5}
    }, torch::kInt64);
    
    // Convert to packed tensors
    auto packed_A = PackedTensorStatic<2, 3>::from_pytorch(torch_A);
    auto packed_B = PackedTensorStatic<3, 2>::from_pytorch(torch_B);
    PackedTensorStatic<2, 2> packed_C;
    
    // Perform GEMM with our packed implementation (automatically clamps to int4)
    packed_gemm_static(packed_A, packed_B, packed_C);
    
    // Simulate int4 inference in PyTorch
    auto torch_C = torch::matmul(torch_A, torch_B);
    // Clamp the result to int4 range (0-15)
    torch_C = torch::clamp(torch_C, 0, 15);
    auto packed_C_torch = packed_C.to_pytorch();
    
    // Both results should now be in int4 range
    EXPECT_TRUE(torch::all(torch_C <= 15).item<bool>());
    EXPECT_TRUE(torch::all(packed_C_torch <= 15).item<bool>());
    
    // And they should match since both are properly clamped
    EXPECT_TRUE(torch::equal(torch_C, packed_C_torch));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 