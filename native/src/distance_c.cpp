#include "vecadvisor/distance_c.h"

#include "vecadvisor/distance.hpp"

#include <algorithm>
#include <queue>
#include <vector>

namespace {

using DistanceKernel = float (*)(const float*, const float*, std::size_t) noexcept;

struct TopKCandidate {
  std::size_t index;
  float distance;
};

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

bool is_better(
    vecadvisor_distance_metric metric,
    const TopKCandidate& candidate,
    const TopKCandidate& incumbent) noexcept {
  if (candidate.distance == incumbent.distance) {
    return candidate.index < incumbent.index;
  }
  if (metric == VECADVISOR_DISTANCE_INNER_PRODUCT) {
    return candidate.distance > incumbent.distance;
  }
  return candidate.distance < incumbent.distance;
}

struct WorstFirst {
  vecadvisor_distance_metric metric;

  bool operator()(const TopKCandidate& left, const TopKCandidate& right) const noexcept {
    return is_better(metric, left, right);
  }
};

float compute_distance_for_topk(
    vecadvisor_distance_metric metric,
    DistanceKernel kernel,
    const float* query,
    float query_norm,
    const float* row,
    std::size_t dim) noexcept {
  if (metric == VECADVISOR_DISTANCE_COSINE) {
    return vecadvisor::native::cosine_distance_with_left_norm(query, query_norm, row, dim);
  }
  return kernel(query, row, dim);
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

vecadvisor_distance_status vecadvisor_distance_topk(
    vecadvisor_distance_metric metric,
    const float* query,
    const float* corpus,
    std::size_t rows,
    std::size_t dim,
    std::size_t k,
    std::size_t* out_indices,
    float* out_distances,
    std::size_t* out_count) {
  if (out_count == nullptr) {
    return VECADVISOR_DISTANCE_INVALID_ARGUMENT;
  }
  *out_count = 0U;
  if (query == nullptr || corpus == nullptr || out_indices == nullptr || out_distances == nullptr ||
      rows == 0U || dim == 0U || k == 0U) {
    return VECADVISOR_DISTANCE_INVALID_ARGUMENT;
  }
  const DistanceKernel kernel = select_kernel(metric);
  if (kernel == nullptr) {
    return VECADVISOR_DISTANCE_UNSUPPORTED_METRIC;
  }

  const std::size_t limit = std::min(k, rows);
  const float query_norm =
      metric == VECADVISOR_DISTANCE_COSINE ? vecadvisor::native::l2_norm(query, dim) : 0.0F;
  std::priority_queue<TopKCandidate, std::vector<TopKCandidate>, WorstFirst> heap{
      WorstFirst{metric}};

  for (std::size_t row = 0; row < rows; ++row) {
    const TopKCandidate candidate{
        row,
        compute_distance_for_topk(metric, kernel, query, query_norm, corpus + row * dim, dim),
    };
    if (heap.size() < limit) {
      heap.push(candidate);
    } else if (is_better(metric, candidate, heap.top())) {
      heap.pop();
      heap.push(candidate);
    }
  }

  std::vector<TopKCandidate> sorted;
  sorted.reserve(heap.size());
  while (!heap.empty()) {
    sorted.push_back(heap.top());
    heap.pop();
  }
  std::sort(sorted.begin(), sorted.end(), [metric](const auto& left, const auto& right) {
    return is_better(metric, left, right);
  });

  for (std::size_t index = 0; index < sorted.size(); ++index) {
    out_indices[index] = sorted[index].index;
    out_distances[index] = sorted[index].distance;
  }
  *out_count = sorted.size();
  return VECADVISOR_DISTANCE_OK;
}
