#pragma once
#include <vector>

// Function to add two vectors on GPU using CUDA
void cudaVectorAdd(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& result);
void cudaVectorAdd(const std::vector<int>& a, const std::vector<int>& b,
                   std::vector<int>& result);
