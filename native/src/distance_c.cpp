#include "vecadvisor/distance_c.h"

#include "vecadvisor/distance.hpp"

namespace {

using DistanceKernel = float (*)(const float*, const float*, std::size_t) noexcept;

DistanceKernel select_kernel(vecadvisor_distance_metric metric) noexcept {
  switch (metric) {
    case VECADVISOR_DISTANCE_L2_SQUARED:
      return vecadvisor::native::l2_squared;
    case VECADVISOR_DISTANCE_INNER_PRODUCT:
      return vecadvisor::native::inner_product;
    case VECADVISOR_DISTANCE_COSINE:
      return vecadvisor::native::cosine_distance;
  }
  return nullptr;
}

bool invalid_vector_args(const float* left, const float* right, std::size_t dim) noexcept {
  return left == nullptr || right == nullptr || dim == 0U;
}

}  // namespace

const char* vecadvisor_distance_status_message(vecadvisor_distance_status status) {
  switch (status) {
    case VECADVISOR_DISTANCE_OK:
      return "ok";
    case VECADVISOR_DISTANCE_INVALID_ARGUMENT:
      return "invalid argument";
    case VECADVISOR_DISTANCE_UNSUPPORTED_METRIC:
      return "unsupported metric";
  }
  return "unknown status";
}

vecadvisor_distance_status vecadvisor_distance_get_capabilities(
    vecadvisor_kernel_capabilities* out) {
  if (out == nullptr) {
    return VECADVISOR_DISTANCE_INVALID_ARGUMENT;
  }
  const auto capabilities = vecadvisor::native::detect_capabilities();
  out->avx2_compiled = capabilities.avx2_compiled ? 1 : 0;
  out->avx2_runtime = capabilities.avx2_runtime ? 1 : 0;
  out->l2_kernel = capabilities.l2_kernel;
  out->inner_product_kernel = capabilities.inner_product_kernel;
  out->cosine_kernel = capabilities.cosine_kernel;
  return VECADVISOR_DISTANCE_OK;
}

vecadvisor_distance_status vecadvisor_distance_compute(
    vecadvisor_distance_metric metric,
    const float* left,
    const float* right,
    std::size_t dim,
    float* out) {
  if (out == nullptr || invalid_vector_args(left, right, dim)) {
    return VECADVISOR_DISTANCE_INVALID_ARGUMENT;
  }
  const DistanceKernel kernel = select_kernel(metric);
  if (kernel == nullptr) {
    return VECADVISOR_DISTANCE_UNSUPPORTED_METRIC;
  }
  *out = kernel(left, right, dim);
  return VECADVISOR_DISTANCE_OK;
}

vecadvisor_distance_status vecadvisor_distance_compute_many(
    vecadvisor_distance_metric metric,
    const float* query,
    const float* corpus,
    std::size_t rows,
    std::size_t dim,
    float* out) {
  if (out == nullptr || query == nullptr || corpus == nullptr || rows == 0U || dim == 0U) {
    return VECADVISOR_DISTANCE_INVALID_ARGUMENT;
  }
  const DistanceKernel kernel = select_kernel(metric);
  if (kernel == nullptr) {
    return VECADVISOR_DISTANCE_UNSUPPORTED_METRIC;
  }
  if (metric == VECADVISOR_DISTANCE_COSINE) {
    const float query_norm = vecadvisor::native::l2_norm(query, dim);
    for (std::size_t row = 0; row < rows; ++row) {
      out[row] = vecadvisor::native::cosine_distance_with_left_norm(
          query, query_norm, corpus + row * dim, dim);
    }
    return VECADVISOR_DISTANCE_OK;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    out[row] = kernel(query, corpus + row * dim, dim);
  }
  return VECADVISOR_DISTANCE_OK;
}
