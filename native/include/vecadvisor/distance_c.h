#pragma once

#include <stddef.h>

#if defined(_WIN32) && defined(VECADVISOR_NATIVE_BUILDING_DLL)
#define VECADVISOR_NATIVE_EXPORT __declspec(dllexport)
#elif defined(_WIN32) && !defined(VECADVISOR_NATIVE_STATIC)
#define VECADVISOR_NATIVE_EXPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define VECADVISOR_NATIVE_EXPORT __attribute__((visibility("default")))
#else
#define VECADVISOR_NATIVE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vecadvisor_distance_status {
  VECADVISOR_DISTANCE_OK = 0,
  VECADVISOR_DISTANCE_INVALID_ARGUMENT = 1,
  VECADVISOR_DISTANCE_UNSUPPORTED_METRIC = 2,
} vecadvisor_distance_status;

typedef enum vecadvisor_distance_metric {
  VECADVISOR_DISTANCE_L2_SQUARED = 1,
  VECADVISOR_DISTANCE_INNER_PRODUCT = 2,
  VECADVISOR_DISTANCE_COSINE = 3,
} vecadvisor_distance_metric;

typedef struct vecadvisor_kernel_capabilities {
  int avx2_compiled;
  int avx2_runtime;
  const char* l2_kernel;
  const char* inner_product_kernel;
  const char* cosine_kernel;
} vecadvisor_kernel_capabilities;

VECADVISOR_NATIVE_EXPORT const char* vecadvisor_distance_status_message(
    vecadvisor_distance_status status);

VECADVISOR_NATIVE_EXPORT vecadvisor_distance_status vecadvisor_distance_get_capabilities(
    vecadvisor_kernel_capabilities* out);

VECADVISOR_NATIVE_EXPORT vecadvisor_distance_status vecadvisor_distance_compute(
    vecadvisor_distance_metric metric,
    const float* left,
    const float* right,
    size_t dim,
    float* out);

VECADVISOR_NATIVE_EXPORT vecadvisor_distance_status vecadvisor_distance_compute_many(
    vecadvisor_distance_metric metric,
    const float* query,
    const float* corpus,
    size_t rows,
    size_t dim,
    float* out);

#ifdef __cplusplus
}
#endif
