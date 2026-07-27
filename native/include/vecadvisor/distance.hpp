#pragma once

#include <cstddef>

namespace vecadvisor::native {

struct KernelCapabilities {
  bool avx2_compiled;
  bool avx2_runtime;
  const char* l2_kernel;
  const char* inner_product_kernel;
  const char* cosine_kernel;
};

float l2_squared_scalar(const float* left, const float* right, std::size_t dim) noexcept;
float inner_product_scalar(const float* left, const float* right, std::size_t dim) noexcept;
float cosine_distance_scalar(const float* left, const float* right, std::size_t dim) noexcept;

float l2_squared(const float* left, const float* right, std::size_t dim) noexcept;
float inner_product(const float* left, const float* right, std::size_t dim) noexcept;
float cosine_distance(const float* left, const float* right, std::size_t dim) noexcept;

KernelCapabilities detect_capabilities() noexcept;

}  // namespace vecadvisor::native
