#include "vecadvisor/distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace vecadvisor::native {

#if VECADVISOR_BUILD_AVX2
float l2_squared_avx2(const float* left, const float* right, std::size_t dim) noexcept;
float inner_product_avx2(const float* left, const float* right, std::size_t dim) noexcept;
float cosine_distance_avx2(const float* left, const float* right, std::size_t dim) noexcept;
#endif

namespace {

constexpr float kMinNorm = 1.0e-12F;

bool cpu_supports_avx2() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int leaf0[4] = {0, 0, 0, 0};
  __cpuidex(leaf0, 0, 0);
  if (leaf0[0] < 7) {
    return false;
  }

  int leaf1[4] = {0, 0, 0, 0};
  __cpuidex(leaf1, 1, 0);
  const bool osxsave = (leaf1[2] & (1 << 27)) != 0;
  const bool avx = (leaf1[2] & (1 << 28)) != 0;
  if (!osxsave || !avx) {
    return false;
  }

  const unsigned long long xcr0 = _xgetbv(0);
  if ((xcr0 & 0x6ULL) != 0x6ULL) {
    return false;
  }

  int leaf7[4] = {0, 0, 0, 0};
  __cpuidex(leaf7, 7, 0);
  return (leaf7[1] & (1 << 5)) != 0;
#elif defined(__x86_64__) || defined(__i386__)
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if (__get_cpuid_max(0, nullptr) < 7) {
    return false;
  }
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  const bool osxsave = (ecx & bit_OSXSAVE) != 0;
  const bool avx = (ecx & bit_AVX) != 0;
  if (!osxsave || !avx) {
    return false;
  }

  std::uint32_t xcr0_eax = 0;
  std::uint32_t xcr0_edx = 0;
#if defined(__GNUC__) || defined(__clang__)
  __asm__ volatile("xgetbv" : "=a"(xcr0_eax), "=d"(xcr0_edx) : "c"(0));
#endif
  const std::uint64_t xcr0 = (static_cast<std::uint64_t>(xcr0_edx) << 32U) | xcr0_eax;
  if ((xcr0 & 0x6ULL) != 0x6ULL) {
    return false;
  }

  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  return (ebx & bit_AVX2) != 0;
#else
  return false;
#endif
}

bool avx2_enabled() noexcept {
#if VECADVISOR_BUILD_AVX2
  static const bool enabled = cpu_supports_avx2();
  return enabled;
#else
  return false;
#endif
}

}  // namespace

float l2_squared_scalar(const float* left, const float* right, std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t index = 0; index < dim; ++index) {
    const float delta = left[index] - right[index];
    sum += delta * delta;
  }
  return sum;
}

float inner_product_scalar(const float* left, const float* right, std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t index = 0; index < dim; ++index) {
    sum += left[index] * right[index];
  }
  return sum;
}

float cosine_distance_scalar(const float* left, const float* right, std::size_t dim) noexcept {
  float dot = 0.0F;
  float left_norm = 0.0F;
  float right_norm = 0.0F;
  for (std::size_t index = 0; index < dim; ++index) {
    const float left_value = left[index];
    const float right_value = right[index];
    dot += left_value * right_value;
    left_norm += left_value * left_value;
    right_norm += right_value * right_value;
  }
  const float denominator = std::max(std::sqrt(left_norm) * std::sqrt(right_norm), kMinNorm);
  return 1.0F - dot / denominator;
}

float l2_squared(const float* left, const float* right, std::size_t dim) noexcept {
#if VECADVISOR_BUILD_AVX2
  if (avx2_enabled()) {
    return l2_squared_avx2(left, right, dim);
  }
#endif
  return l2_squared_scalar(left, right, dim);
}

float inner_product(const float* left, const float* right, std::size_t dim) noexcept {
#if VECADVISOR_BUILD_AVX2
  if (avx2_enabled()) {
    return inner_product_avx2(left, right, dim);
  }
#endif
  return inner_product_scalar(left, right, dim);
}

float cosine_distance(const float* left, const float* right, std::size_t dim) noexcept {
#if VECADVISOR_BUILD_AVX2
  if (avx2_enabled()) {
    return cosine_distance_avx2(left, right, dim);
  }
#endif
  return cosine_distance_scalar(left, right, dim);
}

KernelCapabilities detect_capabilities() noexcept {
  const bool runtime_avx2 = avx2_enabled();
  return KernelCapabilities{
#if VECADVISOR_BUILD_AVX2
      true,
#else
      false,
#endif
      runtime_avx2,
      runtime_avx2 ? "avx2" : "scalar",
      runtime_avx2 ? "avx2" : "scalar",
      runtime_avx2 ? "avx2" : "scalar",
  };
}

}  // namespace vecadvisor::native
