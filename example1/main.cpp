#include <chrono> // For timing
#include <iostream>
#include <numeric> // For std::iota
#include <random>  // For random numbers
#include <vector>

#include "cuda.hpp"
#include "simd.hpp"

// Utility function to print vector (for debugging)
template <typename T>
void printVector(const std::vector<T>& vec, int limit = 10) {
    std::cout << "[";
    for (int i = 0; i < std::min((int)vec.size(), limit); ++i) {
        std::cout << vec[i]
                  << (i == std::min((int)vec.size(), limit) - 1 ? "" : ", ");
    }
    if (vec.size() > limit) {
        std::cout << "...";
    }
    std::cout << "]" << std::endl;
}

// Basic CPU vector addition
template <typename T>
void cpuVectorAdd(const std::vector<T>& a, const std::vector<T>& b,
                  std::vector<T>& result) {
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
}

// Function to verify results
template <typename T>
bool verifyResults(const std::vector<T>& reference, const std::vector<T>& test,
                   double epsilon = 1e-6) {
    if (reference.size() != test.size()) {
        std::cerr << "Verification Error: Vector sizes mismatch." << std::endl;
        return false;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        if constexpr (std::is_floating_point<T>::value) {
            if (std::abs(reference[i] - test[i]) > epsilon) {
                std::cerr << "Verification Error at index " << i
                          << ": Reference " << reference[i] << ", Test "
                          << test[i] << std::endl;
                return false;
            }
        } else {
            if (reference[i] != test[i]) {
                std::cerr << "Verification Error at index " << i
                          << ": Reference " << reference[i] << ", Test "
                          << test[i] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    const size_t N_FLOAT = 250000000; // 25 million elements for floats
    const size_t N_INT = 250000000;   // 25 million elements for ints

    // --- Float Vector Addition ---
    std::cout << "--- Float Vector Addition (N = " << N_FLOAT << ") ---"
              << std::endl;

    std::vector<float> a_float(N_FLOAT);
    std::vector<float> b_float(N_FLOAT);
    std::vector<float> result_cpu_float(N_FLOAT);
    std::vector<float> result_cuda_float(N_FLOAT);
    std::vector<float> result_simd_float(N_FLOAT);

    // Initialize vectors with some data
    std::iota(a_float.begin(), a_float.end(), 1.0f); // 1.0, 2.0, 3.0, ...
    std::iota(b_float.begin(), b_float.end(),
              100.0f); // 100.0, 101.0, 102.0, ...

    // printVector(a_float);
    // printVector(b_float);

    // CPU (Baseline)
    auto start = std::chrono::high_resolution_clock::now();
    cpuVectorAdd(a_float, b_float, result_cpu_float);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU (Float) execution time: " << diff.count() << " s"
              << std::endl;

    // CUDA
    start = std::chrono::high_resolution_clock::now();
    cudaVectorAdd(a_float, b_float, result_cuda_float);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "CUDA (Float) execution time: " << diff.count() << " s"
              << std::endl;
    if (verifyResults(result_cpu_float, result_cuda_float)) {
        std::cout << "CUDA (Float) result verified successfully." << std::endl;
    } else {
        std::cout << "CUDA (Float) result verification FAILED!" << std::endl;
    }

    // SIMD
    start = std::chrono::high_resolution_clock::now();
    simdVectorAdd(a_float, b_float, result_simd_float);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "SIMD (Float) execution time: " << diff.count() << " s"
              << std::endl;
    if (verifyResults(result_cpu_float, result_simd_float)) {
        std::cout << "SIMD (Float) result verified successfully." << std::endl;
    } else {
        std::cout << "SIMD (Float) result verification FAILED!" << std::endl;
    }

    std::cout << std::endl;

    // --- Integer Vector Addition ---
    std::cout << "--- Integer Vector Addition (N = " << N_INT << ") ---"
              << std::endl;

    std::vector<int> a_int(N_INT);
    std::vector<int> b_int(N_INT);
    std::vector<int> result_cpu_int(N_INT);
    std::vector<int> result_cuda_int(N_INT);
    std::vector<int> result_simd_int(N_INT);

    // Initialize vectors with some data
    std::iota(a_int.begin(), a_int.end(), 1);   // 1, 2, 3, ...
    std::iota(b_int.begin(), b_int.end(), 100); // 100, 101, 102, ...

    // CPU (Baseline)
    start = std::chrono::high_resolution_clock::now();
    cpuVectorAdd(a_int, b_int, result_cpu_int);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "CPU (Int) execution time: " << diff.count() << " s"
              << std::endl;

    // CUDA
    start = std::chrono::high_resolution_clock::now();
    cudaVectorAdd(a_int, b_int, result_cuda_int);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "CUDA (Int) execution time: " << diff.count() << " s"
              << std::endl;
    if (verifyResults(result_cpu_int, result_cuda_int)) {
        std::cout << "CUDA (Int) result verified successfully." << std::endl;
    } else {
        std::cout << "CUDA (Int) result verification FAILED!" << std::endl;
    }

    // SIMD
    start = std::chrono::high_resolution_clock::now();
    simdVectorAdd(a_int, b_int, result_simd_int);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "SIMD (Int) execution time: " << diff.count() << " s"
              << std::endl;
    if (verifyResults(result_cpu_int, result_simd_int)) {
        std::cout << "SIMD (Int) result verified successfully." << std::endl;
    } else {
        std::cout << "SIMD (Int) result verification FAILED!" << std::endl;
    }

    return 0;
}
