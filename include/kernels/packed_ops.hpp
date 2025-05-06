#pragma once

#include "packed_tensor.hpp"
#include <cstdint>
#include <vector>
#include <array>
#include <type_traits>
#include <limits>

namespace BitTensor {

namespace detail {
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

// Dynamic GEMM operation for packed tensors
template<typename T>
void packed_gemm(const PackedTensor<T>& A, const PackedTensor<T>& B, PackedTensor<T>& C) {
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
                        
                        uint8_t clamped = static_cast<uint8_t>(std::min<int32_t>(std::max(0, sum), 15));
                        C.set({m, n}, clamped);
                    }
                }
            }
        }
    }
}

// Static GEMM operation for packed tensors
template<int64_t M, int64_t K, int64_t N>
void packed_gemm_static(const PackedTensorStatic<M, K>& A,
                       const PackedTensorStatic<K, N>& B,
                       PackedTensorStatic<M, N>& C) {
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
                        
                        uint8_t clamped = static_cast<uint8_t>(std::min<int32_t>(std::max(0, sum), 15));
                        C.set({m, n}, clamped);
                    }
                }
            }
        }
    }
}

} // namespace BitTensor 