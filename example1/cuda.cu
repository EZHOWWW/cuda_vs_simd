#include "cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for vector addition
// Each thread handles one element addition
__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void addKernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host function to add two vectors on GPU using CUDA
void cudaVectorAdd(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& result) {
    int n = a.size();
    if (n != b.size() || n != result.size()) {
        std::cerr << "Error: Vector sizes mismatch for CUDA float addition."
                  << std::endl;
        return;
    }

    float *dev_a, *dev_b, *dev_c;

    // Allocate GPU memory
    cudaMalloc((void**)&dev_a, n * sizeof(float));
    cudaMalloc((void**)&dev_b, n * sizeof(float));
    cudaMalloc((void**)&dev_c, n * sizeof(float));

    // Copy data from host to GPU
    cudaMemcpy(dev_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    addKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c, n);

    // Copy result from GPU to host
    cudaMemcpy(result.data(), dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void cudaVectorAdd(const std::vector<int>& a, const std::vector<int>& b,
                   std::vector<int>& result) {
    int n = a.size();
    if (n != b.size() || n != result.size()) {
        std::cerr << "Error: Vector sizes mismatch for CUDA int addition."
                  << std::endl;
        return;
    }

    int *dev_a, *dev_b, *dev_c;

    // Allocate GPU memory
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));
    cudaMalloc((void**)&dev_c, n * sizeof(int));

    // Copy data from host to GPU
    cudaMemcpy(dev_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    addKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c, n);

    // Copy result from GPU to host
    cudaMemcpy(result.data(), dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
