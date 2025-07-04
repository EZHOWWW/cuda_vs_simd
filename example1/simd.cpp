#include "simd.hpp"
#include <emmintrin.h> // For SSE2 intrinsics
#include <immintrin.h> // For AVX intrinsics
#include <iostream>

// // Function to add two float vectors using AVX intrinsics
void simdVectorAdd(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& result) {
    int n = a.size();
    if (n != b.size() || n != result.size()) {
        std::cerr << "Error: Vector sizes mismatch for SIMD float addition."
                  << std::endl;
        return;
    }

    // Process elements in chunks of 8 floats (256 bits for AVX)
    int i = 0;
    for (; i + 7 < n; i += 8) {
        // Load 8 floats from a into an AVX register
        __m256 va = _mm256_loadu_ps(&a[i]);
        // Load 8 floats from b into an AVX register
        __m256 vb = _mm256_loadu_ps(&b[i]);
        // Add elements in parallel
        __m256 vc = _mm256_add_ps(va, vb);
        // Store 8 floats back to result
        _mm256_storeu_ps(&result[i], vc);
    }

    // Handle remaining elements (if any)
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

// Function to add two int vectors using SSE2 intrinsics (128-bit)
void simdVectorAdd(const std::vector<int>& a, const std::vector<int>& b,
                   std::vector<int>& result) {
    int n = a.size();
    if (n != b.size() || n != result.size()) {
        std::cerr << "Error: Vector sizes mismatch for SIMD int addition."
                  << std::endl;
        return;
    }

    // Process elements in chunks of 4 ints (128 bits for SSE2)
    int i = 0;
    for (; i + 3 < n; i += 4) {
        // Load 4 ints from a into an SSE2 register
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        // Load 4 ints from b into an SSE2 register
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        // Add elements in parallel
        __m128i vc = _mm_add_epi32(va, vb); // Add packed 32-bit integers
        // Store 4 ints back to result
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&result[i]), vc);
    }

    // Handle remaining elements (if any)
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
