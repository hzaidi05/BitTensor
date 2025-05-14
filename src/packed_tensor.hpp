#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <type_traits>
#include <stdexcept>
#include <limits>

namespace BitTensor {

namespace detail {
    constexpr uint64_t ELEMENTS_PER_WORD = 16; // 64 bits / 4 bits per element

    // Template-based packing of 16 values into a 64-bit word
    template<size_t I = 0>
    constexpr uint64_t pack_16_values_impl(const uint8_t* values, uint64_t result = 0) {
        if constexpr (I < 16) {
            return pack_16_values_impl<I + 1>(
                values,
                result | ((static_cast<uint64_t>(values[I] & 0xF) << (I * 4)))
            );
        }
        return result;
    }

    // Template-based unpacking of 16 values from a 64-bit word
    template<size_t I = 0>
    constexpr void unpack_16_values_impl(uint64_t word, uint8_t* values) {
        if constexpr (I < 16) {
            values[I] = static_cast<uint8_t>((word >> (I * 4)) & 0xF);
            unpack_16_values_impl<I + 1>(word, values);
        }
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

    // Cache-aware computation helpers
    constexpr size_t compute_num_words(size_t num_elements) {
        return (num_elements + ELEMENTS_PER_WORD - 1) / ELEMENTS_PER_WORD;
    }

    constexpr size_t compute_num_elements(size_t num_words) {
        return num_words * ELEMENTS_PER_WORD;
    }

    constexpr size_t compute_cache_line_elements() {
        return 64 / sizeof(uint64_t) * ELEMENTS_PER_WORD;  // 64 bytes per cache line
    }

    constexpr size_t compute_cache_line_words() {
        return 64 / sizeof(uint64_t);  // 64 bytes per cache line
    }
}

template<typename T = uint8_t>
class PackedTensor;

template<typename T>
class PackedTensor {
    static_assert(std::is_integral<T>::value && sizeof(T) == 1,
                 "PackedTensor only supports 8-bit integer types");

public:
    PackedTensor() = default;
    
    PackedTensor(const std::vector<int64_t>& shape);

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

// GEMM operation for packed tensors with optional bias
template<typename T>
void packed_gemm(const PackedTensor<T>& A, const PackedTensor<T>& B, 
                PackedTensor<T>& C, const std::vector<float>* bias = nullptr) {
    static_assert(std::is_integral<T>::value && sizeof(T) == 1,
                 "PackedTensor only supports 8-bit integer types");

    const auto& A_shape = A.shape();
    const auto& B_shape = B.shape();
    const auto& C_shape = C.shape();

    // Check matrix dimensions
    if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }
    if (A_shape[1] != B_shape[0]) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    if (C_shape[0] != A_shape[0] || C_shape[1] != B_shape[1]) {
        throw std::invalid_argument("Output matrix dimensions do not match");
    }
    if (bias && bias->size() != static_cast<size_t>(B_shape[1])) {
        throw std::invalid_argument("Bias size does not match output dimension");
    }

    const int64_t M = A_shape[0];
    const int64_t K = A_shape[1];
    const int64_t N = B_shape[1];

    // Cache-aware blocking
    constexpr int64_t BLOCK_SIZE = static_cast<int64_t>(detail::compute_cache_line_elements());
    const int64_t M_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t K_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t N_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Perform blocked matrix multiplication
    for (int64_t m_block = 0; m_block < M_blocks; ++m_block) {
        const int64_t m_start = m_block * BLOCK_SIZE;
        const int64_t m_end = std::min(m_start + BLOCK_SIZE, M);

        for (int64_t n_block = 0; n_block < N_blocks; ++n_block) {
            const int64_t n_start = n_block * BLOCK_SIZE;
            const int64_t n_end = std::min(n_start + BLOCK_SIZE, N);

            for (int64_t k_block = 0; k_block < K_blocks; ++k_block) {
                const int64_t k_start = k_block * BLOCK_SIZE;
                const int64_t k_end = std::min(k_start + BLOCK_SIZE, K);

                // Process block
                for (int64_t m = m_start; m < m_end; ++m) {
                    for (int64_t n = n_start; n < n_end; ++n) {
                        int32_t sum = 0;
                        for (int64_t k = k_start; k < k_end; ++k) {
                            const int32_t a = static_cast<int32_t>(A({m, k}));
                            const int32_t b = static_cast<int32_t>(B({k, n}));
                            sum += a * b;
                        }
                        
                        // Add bias if provided
                        if (bias) {
                            sum += static_cast<int32_t>((*bias)[n] * 16.0f); // Scale bias to match int4 range
                        }
                        
                        uint8_t clamped = static_cast<uint8_t>(std::min<int32_t>(std::max(0, sum), 15));
                        C.set({m, n}, clamped);
                    }
                }
            }
        }
    }
}

// Static GEMM operation for packed tensors with optional bias
template<int64_t M, int64_t K, int64_t N>
void packed_gemm_static(const PackedTensorStatic<M, K>& A,
                       const PackedTensorStatic<K, N>& B,
                       PackedTensorStatic<M, N>& C,
                       const std::array<float, N>* bias = nullptr) {
    // Cache-aware blocking
    constexpr int64_t BLOCK_SIZE = static_cast<int64_t>(detail::compute_cache_line_elements());
    constexpr int64_t M_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int64_t K_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int64_t N_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Perform blocked matrix multiplication
    for (int64_t m_block = 0; m_block < M_blocks; ++m_block) {
        const int64_t m_start = m_block * BLOCK_SIZE;
        const int64_t m_end = std::min(m_start + BLOCK_SIZE, M);

        for (int64_t n_block = 0; n_block < N_blocks; ++n_block) {
            const int64_t n_start = n_block * BLOCK_SIZE;
            const int64_t n_end = std::min(n_start + BLOCK_SIZE, N);

            for (int64_t k_block = 0; k_block < K_blocks; ++k_block) {
                const int64_t k_start = k_block * BLOCK_SIZE;
                const int64_t k_end = std::min(k_start + BLOCK_SIZE, K);

                // Process block
                for (int64_t m = m_start; m < m_end; ++m) {
                    for (int64_t n = n_start; n < n_end; ++n) {
                        int32_t sum = 0;
                        for (int64_t k = k_start; k < k_end; ++k) {
                            const int32_t a = static_cast<int32_t>(A({m, k}));
                            const int32_t b = static_cast<int32_t>(B({k, n}));
                            sum += a * b;
                        }
                        
                        // Add bias if provided
                        if (bias) {
                            sum += static_cast<int32_t>((*bias)[n] * 16.0f); // Scale bias to match int4 range
                        }
                        
                        uint8_t clamped = static_cast<uint8_t>(std::min<int32_t>(std::max(0, sum), 15));
                        C.set({m, n}, clamped);
                    }
                }
            }
        }
    }
}

} // namespace BitTensor
