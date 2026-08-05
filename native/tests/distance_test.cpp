#include "vecadvisor/distance.hpp"
#include "vecadvisor/distance_c.h"

#include <algorithm>
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

void assert_equal_size(std::size_t actual, std::size_t expected, const char* label) {
  if (actual != expected) {
    std::cerr << label << " expected " << expected << " but got " << actual << '\n';
    std::exit(1);
  }
}

struct ExpectedCandidate {
  std::size_t index;
  float distance;
};

bool expected_better(
    vecadvisor_distance_metric metric,
    const ExpectedCandidate& left,
    const ExpectedCandidate& right) {
  if (left.distance == right.distance) {
    return left.index < right.index;
  }
  if (metric == VECADVISOR_DISTANCE_INNER_PRODUCT) {
    return left.distance > right.distance;
  }
  return left.distance < right.distance;
}

}  // namespace

int main() {
  using vecadvisor::native::cosine_distance;
  using vecadvisor::native::cosine_distance_with_left_norm;
  using vecadvisor::native::cosine_distance_scalar;
  using vecadvisor::native::detect_capabilities;
  using vecadvisor::native::inner_product;
  using vecadvisor::native::inner_product_scalar;
  using vecadvisor::native::l2_norm;
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
  assert_near(l2_norm(left.data(), left.size()), std::sqrt(14.0F), 1.0e-6F, "dispatch norm");
  assert_near(
      cosine_distance_with_left_norm(
          long_left.data(),
          l2_norm(long_left.data(), long_left.size()),
          long_right.data(),
          long_left.size()),
      cosine_distance(long_left.data(), long_right.data(), long_left.size()),
      1.0e-4F,
      "cached-norm cosine");

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

  const std::vector<float> cosine_corpus{0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F};
  if (vecadvisor_distance_compute_many(
          VECADVISOR_DISTANCE_COSINE,
          unit_x.data(),
          cosine_corpus.data(),
          2,
          unit_x.size(),
          many_out.data()) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI batch cosine call failed\n";
    return 1;
  }
  assert_near(many_out[0], 1.0F, 1.0e-6F, "C ABI batch cosine row 0");
  assert_near(many_out[1], 0.0F, 1.0e-6F, "C ABI batch cosine row 1");

  std::vector<std::size_t> topk_indices(5);
  std::vector<float> topk_distances(5);
  std::size_t topk_count = 0;

  const std::vector<float> l2_query{0.0F, 0.0F};
  const std::vector<float> l2_corpus{
      2.0F,
      0.0F,
      1.0F,
      0.0F,
      1.0F,
      0.0F,
      0.0F,
      3.0F,
      0.0F,
      0.0F,
  };
  if (vecadvisor_distance_topk(
          VECADVISOR_DISTANCE_L2_SQUARED,
          l2_query.data(),
          l2_corpus.data(),
          5,
          2,
          3,
          topk_indices.data(),
          topk_distances.data(),
          &topk_count) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI top-k l2 call failed\n";
    return 1;
  }
  assert_equal_size(topk_count, 3, "C ABI top-k l2 count");
  assert_equal_size(topk_indices[0], 4, "C ABI top-k l2 index 0");
  assert_equal_size(topk_indices[1], 1, "C ABI top-k l2 index 1");
  assert_equal_size(topk_indices[2], 2, "C ABI top-k l2 index 2");
  assert_near(topk_distances[0], 0.0F, 1.0e-6F, "C ABI top-k l2 distance 0");
  assert_near(topk_distances[1], 1.0F, 1.0e-6F, "C ABI top-k l2 distance 1");
  assert_near(topk_distances[2], 1.0F, 1.0e-6F, "C ABI top-k l2 distance 2");

  const std::vector<float> ip_query{1.0F, 0.0F};
  const std::vector<float> ip_corpus{
      1.0F,
      0.0F,
      3.0F,
      0.0F,
      3.0F,
      0.0F,
      -1.0F,
      0.0F,
  };
  if (vecadvisor_distance_topk(
          VECADVISOR_DISTANCE_INNER_PRODUCT,
          ip_query.data(),
          ip_corpus.data(),
          4,
          2,
          2,
          topk_indices.data(),
          topk_distances.data(),
          &topk_count) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI top-k inner product call failed\n";
    return 1;
  }
  assert_equal_size(topk_count, 2, "C ABI top-k ip count");
  assert_equal_size(topk_indices[0], 1, "C ABI top-k ip index 0");
  assert_equal_size(topk_indices[1], 2, "C ABI top-k ip index 1");
  assert_near(topk_distances[0], 3.0F, 1.0e-6F, "C ABI top-k ip distance 0");
  assert_near(topk_distances[1], 3.0F, 1.0e-6F, "C ABI top-k ip distance 1");

  const std::vector<float> cosine_topk_corpus{
      0.0F,
      1.0F,
      0.0F,
      1.0F,
      0.0F,
      0.0F,
      -1.0F,
      0.0F,
      0.0F,
      1.0F,
      0.0F,
      0.0F,
  };
  if (vecadvisor_distance_topk(
          VECADVISOR_DISTANCE_COSINE,
          unit_x.data(),
          cosine_topk_corpus.data(),
          4,
          unit_x.size(),
          2,
          topk_indices.data(),
          topk_distances.data(),
          &topk_count) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI top-k cosine call failed\n";
    return 1;
  }
  assert_equal_size(topk_count, 2, "C ABI top-k cosine count");
  assert_equal_size(topk_indices[0], 1, "C ABI top-k cosine index 0");
  assert_equal_size(topk_indices[1], 3, "C ABI top-k cosine index 1");
  assert_near(topk_distances[0], 0.0F, 1.0e-6F, "C ABI top-k cosine distance 0");
  assert_near(topk_distances[1], 0.0F, 1.0e-6F, "C ABI top-k cosine distance 1");

  if (vecadvisor_distance_topk(
          VECADVISOR_DISTANCE_L2_SQUARED,
          l2_query.data(),
          l2_corpus.data(),
          5,
          2,
          9,
          topk_indices.data(),
          topk_distances.data(),
          &topk_count) != VECADVISOR_DISTANCE_OK) {
    std::cerr << "C ABI top-k k>rows call failed\n";
    return 1;
  }
  assert_equal_size(topk_count, 5, "C ABI top-k k>rows count");

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

  if (vecadvisor_distance_topk(
          VECADVISOR_DISTANCE_L2_SQUARED,
          nullptr,
          l2_corpus.data(),
          5,
          2,
          3,
          topk_indices.data(),
          topk_distances.data(),
          &topk_count) != VECADVISOR_DISTANCE_INVALID_ARGUMENT) {
    std::cerr << "C ABI top-k null pointer should fail\n";
    return 1;
  }

  if (vecadvisor_distance_topk(
          static_cast<vecadvisor_distance_metric>(999),
          l2_query.data(),
          l2_corpus.data(),
          5,
          2,
          3,
          topk_indices.data(),
          topk_distances.data(),
          &topk_count) != VECADVISOR_DISTANCE_UNSUPPORTED_METRIC) {
    std::cerr << "C ABI top-k unsupported metric should fail\n";
    return 1;
  }

  const std::size_t reference_rows = 11;
  const std::size_t reference_dim = 31;
  const std::size_t reference_k = 4;
  std::vector<float> reference_query(reference_dim);
  std::vector<float> reference_corpus(reference_rows * reference_dim);
  for (std::size_t dim = 0; dim < reference_dim; ++dim) {
    reference_query[dim] = static_cast<float>(static_cast<int>(dim % 7) - 3) * 0.125F;
  }
  for (std::size_t row = 0; row < reference_rows; ++row) {
    for (std::size_t dim = 0; dim < reference_dim; ++dim) {
      reference_corpus[row * reference_dim + dim] =
          static_cast<float>(((row + 1) * (dim + 3)) % 17) * 0.05F;
    }
  }

  for (const auto metric :
       {VECADVISOR_DISTANCE_L2_SQUARED,
        VECADVISOR_DISTANCE_INNER_PRODUCT,
        VECADVISOR_DISTANCE_COSINE}) {
    std::vector<ExpectedCandidate> expected;
    expected.reserve(reference_rows);
    for (std::size_t row = 0; row < reference_rows; ++row) {
      float distance = 0.0F;
      if (vecadvisor_distance_compute(
              metric,
              reference_query.data(),
              reference_corpus.data() + row * reference_dim,
              reference_dim,
              &distance) != VECADVISOR_DISTANCE_OK) {
        std::cerr << "C ABI reference distance call failed\n";
        return 1;
      }
      expected.push_back(ExpectedCandidate{row, distance});
    }
    std::sort(expected.begin(), expected.end(), [metric](const auto& left, const auto& right) {
      return expected_better(metric, left, right);
    });

    if (vecadvisor_distance_topk(
            metric,
            reference_query.data(),
            reference_corpus.data(),
            reference_rows,
            reference_dim,
            reference_k,
            topk_indices.data(),
            topk_distances.data(),
            &topk_count) != VECADVISOR_DISTANCE_OK) {
      std::cerr << "C ABI reference top-k call failed\n";
      return 1;
    }
    assert_equal_size(topk_count, reference_k, "C ABI reference top-k count");
    for (std::size_t index = 0; index < reference_k; ++index) {
      assert_equal_size(topk_indices[index], expected[index].index, "C ABI reference top-k index");
      assert_near(
          topk_distances[index],
          expected[index].distance,
          1.0e-4F,
          "C ABI reference top-k distance");
    }
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
