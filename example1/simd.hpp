#pragma once
#include <vector>

// Function to add two vectors using SIMD (AVX/SSE) instructions
void simdVectorAdd(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& result);
void simdVectorAdd(const std::vector<int>& a, const std::vector<int>& b,
                   std::vector<int>& result);
