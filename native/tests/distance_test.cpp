#include "vecadvisor/distance.hpp"
#include "vecadvisor/distance_c.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void assert_near(float actual, float expected, float tolerance, const char* label) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << label << " expected " << expected << " but got " << actual << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  using vecadvisor::native::cosine_distance;
  using vecadvisor::native::cosine_distance_scalar;
  using vecadvisor::native::detect_capabilities;
  using vecadvisor::native::inner_product;
  using vecadvisor::native::inner_product_scalar;
  using vecadvisor::native::l2_squared;
  using vecadvisor::native::l2_squared_scalar;

  const std::vector<float> left{1.0F, 2.0F, 3.0F};
  const std::vector<float> right{1.0F, 4.0F, 1.0F};
  assert_near(l2_squared_scalar(left.data(), right.data(), left.size()), 8.0F, 1.0e-6F, "l2");
  assert_near(
      inner_product_scalar(left.data(), right.data(), left.size()), 12.0F, 1.0e-6F, "ip");

  const std::vector<float> unit_x{1.0F, 0.0F, 0.0F};
  const std::vector<float> unit_y{0.0F, 1.0F, 0.0F};
  assert_near(
      cosine_distance_scalar(unit_x.data(), unit_x.data(), unit_x.size()),
      0.0F,
      1.0e-6F,
      "cosine identical");
  assert_near(
      cosine_distance_scalar(unit_x.data(), unit_y.data(), unit_x.size()),
      1.0F,
      1.0e-6F,
      "cosine orthogonal");

  std::vector<float> long_left(37);
  std::vector<float> long_right(37);
  for (std::size_t index = 0; index < long_left.size(); ++index) {
    long_left[index] = static_cast<float>(index % 11) * 0.25F;
    long_right[index] = static_cast<float>((index + 3) % 7) * -0.5F;
  }

  assert_near(
      l2_squared(long_left.data(), long_right.data(), long_left.size()),
      l2_squared_scalar(long_left.data(), long_right.data(), long_left.size()),
      1.0e-4F,
      "dispatch l2");
  assert_near(
      inner_product(long_left.data(), long_right.data(), long_left.size()),
      inner_product_scalar(long_left.data(), long_right.data(), long_left.size()),
      1.0e-4F,
      "dispatch ip");
  assert_near(
      cosine_distance(long_left.data(), long_right.data(), long_left.size()),
      cosine_distance_scalar(long_left.data(), long_right.data(), long_left.size()),
      1.0e-4F,
      "dispatch cosine");

  float c_distance = -1.0F;
  if (vecadvisor_distance_compute(
          VECADVISOR_DISTANCE_L2_SQUARED,
          left.data(),
          right.data(),
          left.size(),
          &c_distance) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI l2 call failed\n";
    return 1;
  }
  assert_near(c_distance, 8.0F, 1.0e-6F, "C ABI l2");

  std::vector<float> many_out(2);
  const std::vector<float> corpus{1.0F, 4.0F, 1.0F, 1.0F, 0.0F, 0.0F};
  if (vecadvisor_distance_compute_many(
          VECADVISOR_DISTANCE_L2_SQUARED,
          left.data(),
          corpus.data(),
          2,
          left.size(),
          many_out.data()) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI batch l2 call failed\n";
    return 1;
  }
  assert_near(many_out[0], 8.0F, 1.0e-6F, "C ABI batch l2 row 0");
  assert_near(many_out[1], 13.0F, 1.0e-6F, "C ABI batch l2 row 1");

  if (vecadvisor_distance_compute(
          static_cast<vecadvisor_distance_metric>(999),
          left.data(),
          right.data(),
          left.size(),
          &c_distance) != VECADVISOR_DISTANCE_UNSUPPORTED_METRIC) {
    std::cerr << "C ABI unsupported metric should fail\n";
    return 1;
  }

  if (vecadvisor_distance_compute(
          VECADVISOR_DISTANCE_L2_SQUARED,
          nullptr,
          right.data(),
          left.size(),
          &c_distance) != VECADVISOR_DISTANCE_INVALID_ARGUMENT) {
    std::cerr << "C ABI null pointer should fail\n";
    return 1;
  }

  const auto capabilities = detect_capabilities();
  if (capabilities.l2_kernel == nullptr || capabilities.inner_product_kernel == nullptr ||
      capabilities.cosine_kernel == nullptr) {
    std::cerr << "kernel names must not be null\n";
    return 1;
  }
  std::cout << "VecAdvisor native kernels: l2=" << capabilities.l2_kernel
            << " ip=" << capabilities.inner_product_kernel
            << " cosine=" << capabilities.cosine_kernel
            << " avx2_compiled=" << capabilities.avx2_compiled
            << " avx2_runtime=" << capabilities.avx2_runtime << '\n';

  vecadvisor_kernel_capabilities c_capabilities{};
  if (vecadvisor_distance_get_capabilities(&c_capabilities) != VECADVISOR_DISTANCE_OK ||
      c_capabilities.l2_kernel == nullptr || c_capabilities.inner_product_kernel == nullptr ||
      c_capabilities.cosine_kernel == nullptr) {
    std::cerr << "C ABI capabilities call failed\n";
    return 1;
  }
  return 0;
}
