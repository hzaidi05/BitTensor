#include <gtest/gtest.h>
#include <packed_tensor.hpp>
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 