#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <type_traits>
#include <memory>
#include <torch/torch.h>
#include <stdexcept>

namespace BitTensor {

namespace detail {
    constexpr uint64_t ELEMENTS_PER_WORD = 16; // 64 bits / 4 bits per element

    // Helper functions for bit manipulation
    constexpr uint8_t extract_4bit(uint64_t word, size_t pos) {
        return static_cast<uint8_t>((word >> (pos * 4)) & 0xF);
    }

    constexpr uint64_t insert_4bit(uint64_t word, uint8_t value, size_t pos) {
        const uint64_t mask = ~(0xFULL << (pos * 4));
        return (word & mask) | ((static_cast<uint64_t>(value) & 0xF) << (pos * 4));
    }

    // Helper functions for bounds checking
    inline void check_bounds(const std::vector<int64_t>& indices, const std::vector<int64_t>& shape) {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Number of indices does not match tensor rank");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    template<size_t N>
    inline void check_bounds(const std::array<int64_t, N>& indices, const std::array<int64_t, N>& shape) {
        for (size_t i = 0; i < N; ++i) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    inline void check_linear_bounds(size_t idx, size_t size) {
        if (idx >= size) {
            throw std::out_of_range("Linear index out of bounds");
        }
    }
}

// Forward declaration
template<typename T = uint8_t>
class PackedTensor;

// Template class definition
template<typename T>
class PackedTensor {
    static_assert(std::is_integral<T>::value && sizeof(T) == 1,
                 "PackedTensor only supports 8-bit integer types");

public:
    PackedTensor() = default;
    
    // Create from shape
    PackedTensor(const std::vector<int64_t>& shape);
    
    // Create from PyTorch tensor
    static PackedTensor from_pytorch(const torch::Tensor& tensor);
    
    // Convert to PyTorch tensor
    torch::Tensor to_pytorch() const;

    T operator()(const std::vector<int64_t>& indices) const;
    void set(const std::vector<int64_t>& indices, T value);
    
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t size() const { return size_; }
    int64_t numel() const { return numel_; }

    const uint64_t* data() const { return data_.data(); }
    uint64_t* data() { return data_.data(); }

    // Linear access methods
    T get_linear(size_t idx) const;
    void set_linear(size_t idx, T value);

private:
    size_t compute_linear_index(const std::vector<int64_t>& indices) const;
    size_t compute_word_index(size_t linear_index) const;
    size_t compute_bit_position(size_t linear_index) const;

    std::vector<int64_t> shape_;
    int64_t size_;  // Total number of elements
    int64_t numel_; // Total number of packed words
    std::vector<uint64_t> data_; // Packed data storage
};

template<typename T>
PackedTensor<T>::PackedTensor(const std::vector<int64_t>& shape)
    : shape_(shape) {
    size_ = 1;
    for (const auto& dim : shape_) {
        size_ *= dim;
    }
    numel_ = (size_ + detail::ELEMENTS_PER_WORD - 1) / detail::ELEMENTS_PER_WORD;
    data_.resize(numel_);
}

template<typename T>
T PackedTensor<T>::operator()(const std::vector<int64_t>& indices) const {
    detail::check_bounds(indices, shape_);
    const size_t linear_idx = compute_linear_index(indices);
    return get_linear(linear_idx);
}

template<typename T>
void PackedTensor<T>::set(const std::vector<int64_t>& indices, T value) {
    detail::check_bounds(indices, shape_);
    const size_t linear_idx = compute_linear_index(indices);
    set_linear(linear_idx, value);
}

template<typename T>
size_t PackedTensor<T>::compute_linear_index(const std::vector<int64_t>& indices) const {
    size_t idx = 0;
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= shape_[i];
    }
    return idx;
}

template<typename T>
size_t PackedTensor<T>::compute_word_index(size_t linear_index) const {
    return linear_index / detail::ELEMENTS_PER_WORD;
}

template<typename T>
size_t PackedTensor<T>::compute_bit_position(size_t linear_index) const {
    return linear_index % detail::ELEMENTS_PER_WORD;
}

template<typename T>
PackedTensor<T> PackedTensor<T>::from_pytorch(const torch::Tensor& tensor) {
    if (tensor.dtype() != torch::kInt64) {
        throw std::invalid_argument("Input tensor must be of type int64");
    }
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        shape.push_back(tensor.size(i));
    }
    PackedTensor<T> result(shape);
    auto tensor_flat = tensor.flatten();
    for (int64_t i = 0; i < tensor_flat.size(0); ++i) {
        result.set_linear(i, static_cast<T>(tensor_flat[i].item<int64_t>()));
    }
    return result;
}

template<typename T>
torch::Tensor PackedTensor<T>::to_pytorch() const {
    auto result = torch::empty(shape_, torch::kInt64);
    auto result_flat = result.flatten();
    for (int64_t i = 0; i < size_; ++i) {
        result_flat[i] = static_cast<int64_t>(get_linear(i));
    }
    return result;
}

// Add helper methods for linear access
template<typename T>
T PackedTensor<T>::get_linear(size_t idx) const {
    detail::check_linear_bounds(idx, size_);
    const size_t word_idx = compute_word_index(idx);
    const size_t bit_pos = compute_bit_position(idx);
    return static_cast<T>(detail::extract_4bit(data_[word_idx], bit_pos));
}

template<typename T>
void PackedTensor<T>::set_linear(size_t idx, T value) {
    detail::check_linear_bounds(idx, size_);
    const size_t word_idx = compute_word_index(idx);
    const size_t bit_pos = compute_bit_position(idx);
    data_[word_idx] = detail::insert_4bit(data_[word_idx], value, bit_pos);
}

// Static tensor implementation
template<int64_t... Dims>
class PackedTensorStatic {
    static_assert(sizeof...(Dims) > 0, "PackedTensorStatic must have at least one dimension");
    static_assert((std::is_convertible_v<decltype(Dims), int64_t> && ...), "All dimensions must be convertible to int64_t");

public:
    static constexpr size_t rank = sizeof...(Dims);
    static constexpr std::array<int64_t, rank> shape = {Dims...};
    static constexpr int64_t size = (Dims * ...);
    static constexpr int64_t numel = (size + detail::ELEMENTS_PER_WORD - 1) / detail::ELEMENTS_PER_WORD;

    PackedTensorStatic() : data_(numel, 0) {}

    // Create from PyTorch tensor
    static PackedTensorStatic from_pytorch(const torch::Tensor& tensor) {
        if (tensor.dtype() != torch::kInt64) {
            throw std::invalid_argument("Input tensor must be of type int64");
        }
        if (tensor.dim() != rank) {
            throw std::invalid_argument("Input tensor rank does not match PackedTensorStatic rank");
        }
        for (size_t i = 0; i < rank; ++i) {
            if (tensor.size(i) != shape[i]) {
                throw std::invalid_argument("Input tensor shape does not match PackedTensorStatic shape");
            }
        }
        
        PackedTensorStatic result;
        auto tensor_flat = tensor.flatten();
        for (int64_t i = 0; i < size; ++i) {
            result.set_linear(i, static_cast<uint8_t>(tensor_flat[i].item<int64_t>()));
        }
        return result;
    }
    
    // Convert to PyTorch tensor
    torch::Tensor to_pytorch() const {
        auto result = torch::empty(shape, torch::kInt64);
        auto result_flat = result.flatten();
        for (int64_t i = 0; i < size; ++i) {
            result_flat[i] = static_cast<int64_t>(get_linear(i));
        }
        return result;
    }

    // Shape information accessors
    static constexpr const std::array<int64_t, rank>& get_shape() { return shape; }
    static constexpr int64_t get_size() { return size; }
    static constexpr int64_t get_numel() { return numel; }

    uint8_t operator()(const std::array<int64_t, rank>& indices) const {
        detail::check_bounds(indices, shape);
        const size_t linear_idx = compute_linear_index(indices);
        const size_t word_idx = compute_word_index(linear_idx);
        const size_t bit_pos = compute_bit_position(linear_idx);
        return detail::extract_4bit(data_[word_idx], bit_pos);
    }

    void set(const std::array<int64_t, rank>& indices, uint8_t value) {
        detail::check_bounds(indices, shape);
        const size_t linear_idx = compute_linear_index(indices);
        const size_t word_idx = compute_word_index(linear_idx);
        const size_t bit_pos = compute_bit_position(linear_idx);
        data_[word_idx] = detail::insert_4bit(data_[word_idx], value, bit_pos);
    }

    // Linear access methods
    uint8_t get_linear(size_t idx) const {
        detail::check_linear_bounds(idx, size);
        const size_t word_idx = compute_word_index(idx);
        const size_t bit_pos = compute_bit_position(idx);
        return detail::extract_4bit(data_[word_idx], bit_pos);
    }

    void set_linear(size_t idx, uint8_t value) {
        detail::check_linear_bounds(idx, size);
        const size_t word_idx = compute_word_index(idx);
        const size_t bit_pos = compute_bit_position(idx);
        data_[word_idx] = detail::insert_4bit(data_[word_idx], value, bit_pos);
    }

    const uint64_t* data() const { return data_.data(); }
    uint64_t* data() { return data_.data(); }

private:
    size_t compute_linear_index(const std::array<int64_t, rank>& indices) const {
        size_t idx = 0;
        size_t stride = 1;
        for (size_t i = 0; i < rank; ++i) {
            idx += indices[rank - 1 - i] * stride;
            stride *= shape[rank - 1 - i];
        }
        return idx;
    }

    static constexpr size_t compute_word_index(size_t linear_index) {
        return linear_index / detail::ELEMENTS_PER_WORD;
    }

    static constexpr size_t compute_bit_position(size_t linear_index) {
        return linear_index % detail::ELEMENTS_PER_WORD;
    }

    std::vector<uint64_t> data_;
};

} // namespace BitTensor
