#include "vecadvisor/distance.hpp"

#include <algorithm>
#include <cmath>

#include <immintrin.h>

namespace vecadvisor::native {

namespace {

constexpr float kMinNorm = 1.0e-12F;

float horizontal_sum(__m256 value) noexcept {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  __m128 shuffle = _mm_movehdup_ps(sum);
  sum = _mm_add_ps(sum, shuffle);
  shuffle = _mm_movehl_ps(shuffle, sum);
  sum = _mm_add_ss(sum, shuffle);
  return _mm_cvtss_f32(sum);
}

}  // namespace

float l2_squared_avx2(const float* left, const float* right, std::size_t dim) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    const __m256 delta = _mm256_sub_ps(left_values, right_values);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(delta, delta));
  }

  float sum = horizontal_sum(acc);
  for (; index < dim; ++index) {
    const float delta = left[index] - right[index];
    sum += delta * delta;
  }
  return sum;
}

float inner_product_avx2(const float* left, const float* right, std::size_t dim) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(left_values, right_values));
  }

  float sum = horizontal_sum(acc);
  for (; index < dim; ++index) {
    sum += left[index] * right[index];
  }
  return sum;
}

float cosine_distance_avx2(const float* left, const float* right, std::size_t dim) noexcept {
  __m256 dot_acc = _mm256_setzero_ps();
  __m256 left_acc = _mm256_setzero_ps();
  __m256 right_acc = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(left_values, right_values));
    left_acc = _mm256_add_ps(left_acc, _mm256_mul_ps(left_values, left_values));
    right_acc = _mm256_add_ps(right_acc, _mm256_mul_ps(right_values, right_values));
  }

  float dot = horizontal_sum(dot_acc);
  float left_norm = horizontal_sum(left_acc);
  float right_norm = horizontal_sum(right_acc);
  for (; index < dim; ++index) {
    const float left_value = left[index];
    const float right_value = right[index];
    dot += left_value * right_value;
    left_norm += left_value * left_value;
    right_norm += right_value * right_value;
  }

  const float denominator = std::max(std::sqrt(left_norm) * std::sqrt(right_norm), kMinNorm);
  return 1.0F - dot / denominator;
}

}  // namespace vecadvisor::native
