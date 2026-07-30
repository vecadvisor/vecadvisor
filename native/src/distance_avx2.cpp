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

float horizontal_sum4(__m256 acc0, __m256 acc1, __m256 acc2, __m256 acc3) noexcept {
  return horizontal_sum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

}  // namespace

float l2_squared_avx2(const float* left, const float* right, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 32 <= dim; index += 32) {
    const __m256 left0 = _mm256_loadu_ps(left + index);
    const __m256 right0 = _mm256_loadu_ps(right + index);
    const __m256 delta0 = _mm256_sub_ps(left0, right0);
    acc0 = _mm256_fmadd_ps(delta0, delta0, acc0);

    const __m256 left1 = _mm256_loadu_ps(left + index + 8);
    const __m256 right1 = _mm256_loadu_ps(right + index + 8);
    const __m256 delta1 = _mm256_sub_ps(left1, right1);
    acc1 = _mm256_fmadd_ps(delta1, delta1, acc1);

    const __m256 left2 = _mm256_loadu_ps(left + index + 16);
    const __m256 right2 = _mm256_loadu_ps(right + index + 16);
    const __m256 delta2 = _mm256_sub_ps(left2, right2);
    acc2 = _mm256_fmadd_ps(delta2, delta2, acc2);

    const __m256 left3 = _mm256_loadu_ps(left + index + 24);
    const __m256 right3 = _mm256_loadu_ps(right + index + 24);
    const __m256 delta3 = _mm256_sub_ps(left3, right3);
    acc3 = _mm256_fmadd_ps(delta3, delta3, acc3);
  }

  __m256 tail_acc = _mm256_setzero_ps();
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    const __m256 delta = _mm256_sub_ps(left_values, right_values);
    tail_acc = _mm256_fmadd_ps(delta, delta, tail_acc);
  }

  float sum = horizontal_sum4(acc0, acc1, acc2, acc3) + horizontal_sum(tail_acc);
  for (; index < dim; ++index) {
    const float delta = left[index] - right[index];
    sum += delta * delta;
  }
  return sum;
}

float inner_product_avx2(const float* left, const float* right, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 32 <= dim; index += 32) {
    const __m256 left0 = _mm256_loadu_ps(left + index);
    const __m256 right0 = _mm256_loadu_ps(right + index);
    acc0 = _mm256_fmadd_ps(left0, right0, acc0);

    const __m256 left1 = _mm256_loadu_ps(left + index + 8);
    const __m256 right1 = _mm256_loadu_ps(right + index + 8);
    acc1 = _mm256_fmadd_ps(left1, right1, acc1);

    const __m256 left2 = _mm256_loadu_ps(left + index + 16);
    const __m256 right2 = _mm256_loadu_ps(right + index + 16);
    acc2 = _mm256_fmadd_ps(left2, right2, acc2);

    const __m256 left3 = _mm256_loadu_ps(left + index + 24);
    const __m256 right3 = _mm256_loadu_ps(right + index + 24);
    acc3 = _mm256_fmadd_ps(left3, right3, acc3);
  }

  __m256 tail_acc = _mm256_setzero_ps();
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    tail_acc = _mm256_fmadd_ps(left_values, right_values, tail_acc);
  }

  float sum = horizontal_sum4(acc0, acc1, acc2, acc3) + horizontal_sum(tail_acc);
  for (; index < dim; ++index) {
    sum += left[index] * right[index];
  }
  return sum;
}

float cosine_distance_avx2(const float* left, const float* right, std::size_t dim) noexcept {
  __m256 dot0 = _mm256_setzero_ps();
  __m256 dot1 = _mm256_setzero_ps();
  __m256 dot2 = _mm256_setzero_ps();
  __m256 dot3 = _mm256_setzero_ps();
  __m256 left0_acc = _mm256_setzero_ps();
  __m256 left1_acc = _mm256_setzero_ps();
  __m256 left2_acc = _mm256_setzero_ps();
  __m256 left3_acc = _mm256_setzero_ps();
  __m256 right0_acc = _mm256_setzero_ps();
  __m256 right1_acc = _mm256_setzero_ps();
  __m256 right2_acc = _mm256_setzero_ps();
  __m256 right3_acc = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 32 <= dim; index += 32) {
    const __m256 left0 = _mm256_loadu_ps(left + index);
    const __m256 right0 = _mm256_loadu_ps(right + index);
    dot0 = _mm256_fmadd_ps(left0, right0, dot0);
    left0_acc = _mm256_fmadd_ps(left0, left0, left0_acc);
    right0_acc = _mm256_fmadd_ps(right0, right0, right0_acc);

    const __m256 left1 = _mm256_loadu_ps(left + index + 8);
    const __m256 right1 = _mm256_loadu_ps(right + index + 8);
    dot1 = _mm256_fmadd_ps(left1, right1, dot1);
    left1_acc = _mm256_fmadd_ps(left1, left1, left1_acc);
    right1_acc = _mm256_fmadd_ps(right1, right1, right1_acc);

    const __m256 left2 = _mm256_loadu_ps(left + index + 16);
    const __m256 right2 = _mm256_loadu_ps(right + index + 16);
    dot2 = _mm256_fmadd_ps(left2, right2, dot2);
    left2_acc = _mm256_fmadd_ps(left2, left2, left2_acc);
    right2_acc = _mm256_fmadd_ps(right2, right2, right2_acc);

    const __m256 left3 = _mm256_loadu_ps(left + index + 24);
    const __m256 right3 = _mm256_loadu_ps(right + index + 24);
    dot3 = _mm256_fmadd_ps(left3, right3, dot3);
    left3_acc = _mm256_fmadd_ps(left3, left3, left3_acc);
    right3_acc = _mm256_fmadd_ps(right3, right3, right3_acc);
  }

  __m256 dot_tail = _mm256_setzero_ps();
  __m256 left_tail = _mm256_setzero_ps();
  __m256 right_tail = _mm256_setzero_ps();
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    dot_tail = _mm256_fmadd_ps(left_values, right_values, dot_tail);
    left_tail = _mm256_fmadd_ps(left_values, left_values, left_tail);
    right_tail = _mm256_fmadd_ps(right_values, right_values, right_tail);
  }

  float dot = horizontal_sum4(dot0, dot1, dot2, dot3) + horizontal_sum(dot_tail);
  float left_norm = horizontal_sum4(left0_acc, left1_acc, left2_acc, left3_acc) +
                    horizontal_sum(left_tail);
  float right_norm = horizontal_sum4(right0_acc, right1_acc, right2_acc, right3_acc) +
                     horizontal_sum(right_tail);
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

float l2_norm_avx2(const float* value, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 32 <= dim; index += 32) {
    const __m256 value0 = _mm256_loadu_ps(value + index);
    const __m256 value1 = _mm256_loadu_ps(value + index + 8);
    const __m256 value2 = _mm256_loadu_ps(value + index + 16);
    const __m256 value3 = _mm256_loadu_ps(value + index + 24);
    acc0 = _mm256_fmadd_ps(value0, value0, acc0);
    acc1 = _mm256_fmadd_ps(value1, value1, acc1);
    acc2 = _mm256_fmadd_ps(value2, value2, acc2);
    acc3 = _mm256_fmadd_ps(value3, value3, acc3);
  }

  __m256 tail_acc = _mm256_setzero_ps();
  for (; index + 8 <= dim; index += 8) {
    const __m256 values = _mm256_loadu_ps(value + index);
    tail_acc = _mm256_fmadd_ps(values, values, tail_acc);
  }

  float sum = horizontal_sum4(acc0, acc1, acc2, acc3) + horizontal_sum(tail_acc);
  for (; index < dim; ++index) {
    sum += value[index] * value[index];
  }
  return std::sqrt(sum);
}

float cosine_distance_with_left_norm_avx2(
    const float* left,
    float left_norm,
    const float* right,
    std::size_t dim) noexcept {
  __m256 dot0 = _mm256_setzero_ps();
  __m256 dot1 = _mm256_setzero_ps();
  __m256 dot2 = _mm256_setzero_ps();
  __m256 dot3 = _mm256_setzero_ps();
  __m256 right0_acc = _mm256_setzero_ps();
  __m256 right1_acc = _mm256_setzero_ps();
  __m256 right2_acc = _mm256_setzero_ps();
  __m256 right3_acc = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 32 <= dim; index += 32) {
    const __m256 left0 = _mm256_loadu_ps(left + index);
    const __m256 right0 = _mm256_loadu_ps(right + index);
    dot0 = _mm256_fmadd_ps(left0, right0, dot0);
    right0_acc = _mm256_fmadd_ps(right0, right0, right0_acc);

    const __m256 left1 = _mm256_loadu_ps(left + index + 8);
    const __m256 right1 = _mm256_loadu_ps(right + index + 8);
    dot1 = _mm256_fmadd_ps(left1, right1, dot1);
    right1_acc = _mm256_fmadd_ps(right1, right1, right1_acc);

    const __m256 left2 = _mm256_loadu_ps(left + index + 16);
    const __m256 right2 = _mm256_loadu_ps(right + index + 16);
    dot2 = _mm256_fmadd_ps(left2, right2, dot2);
    right2_acc = _mm256_fmadd_ps(right2, right2, right2_acc);

    const __m256 left3 = _mm256_loadu_ps(left + index + 24);
    const __m256 right3 = _mm256_loadu_ps(right + index + 24);
    dot3 = _mm256_fmadd_ps(left3, right3, dot3);
    right3_acc = _mm256_fmadd_ps(right3, right3, right3_acc);
  }

  __m256 dot_tail = _mm256_setzero_ps();
  __m256 right_tail = _mm256_setzero_ps();
  for (; index + 8 <= dim; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    dot_tail = _mm256_fmadd_ps(left_values, right_values, dot_tail);
    right_tail = _mm256_fmadd_ps(right_values, right_values, right_tail);
  }

  float dot = horizontal_sum4(dot0, dot1, dot2, dot3) + horizontal_sum(dot_tail);
  float right_norm = horizontal_sum4(right0_acc, right1_acc, right2_acc, right3_acc) +
                     horizontal_sum(right_tail);
  for (; index < dim; ++index) {
    const float left_value = left[index];
    const float right_value = right[index];
    dot += left_value * right_value;
    right_norm += right_value * right_value;
  }

  const float denominator = std::max(left_norm * std::sqrt(right_norm), kMinNorm);
  return 1.0F - dot / denominator;
}

}  // namespace vecadvisor::native
