# Native Distance Kernel Benchmark

This committed MVP2 artifact compares the normal runtime-dispatch build against an explicit scalar-only fallback build.

- rows: `2048`
- queries: `8`
- dimensions: `128`
- iterations: `5`
- AVX2 run platform: `Linux-6.17.0-1020-azure-x86_64-with-glibc2.39`
- scalar fallback platform: `Linux-6.17.0-1020-azure-x86_64-with-glibc2.39`
- AVX2 runtime available: `True`
- scalar build compiled AVX2: `False`

| metric | AVX2 kernel | AVX2 ns/dist | SimSIMD ns/dist | AVX2/SimSIMD | scalar fallback ns/dist | speedup | max AVX2 error | checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| l2 | avx2 | 15.760 | 20.548 | 0.767x | 83.690 | 5.310x | 5.34058e-05 | pass / simsimd:pass |
| ip | avx2 | 14.350 | 20.434 | 0.702x | 77.367 | 5.392x | 4.76837e-06 | pass / simsimd:pass |
| cosine | avx2 | 23.827 | 30.848 | 0.772x | 120.841 | 5.072x | 1.19209e-07 | pass / simsimd:pass |

## Interpretation

The `AVX2 ns/dist` column is the runtime-dispatch path future Python bindings would call on an AVX2-capable host. The scalar fallback column comes from a separate build configured with `-DVECADVISOR_NATIVE_ENABLE_AVX2=OFF`, so it verifies the portable path used on machines without AVX2 support.

`SimSIMD ns/dist` is an optional external baseline measured by the Python benchmark wrapper on the same deterministic matrices. `AVX2/SimSIMD` is a latency ratio; lower is better for VecAdvisor.

## Reproduce

```bash
python tools/native_distance_benchmark.py \
  --build-dir native/build-avx2 \
  --rows 2048 \
  --queries 8 \
  --dim 128 \
  --iterations 5 \
  --external-baselines simsimd \
  --json-out native/build-avx2/native-distance-kernels.json \
  --markdown-out native/build-avx2/native-distance-kernels.md \
  --svg-out native/build-avx2/native-distance-kernels.svg

python tools/native_distance_benchmark.py \
  --build-dir native/build-scalar \
  --disable-avx2 \
  --rows 2048 \
  --queries 8 \
  --dim 128 \
  --iterations 5 \
  --json-out native/build-scalar/native-distance-kernels.json \
  --markdown-out native/build-scalar/native-distance-kernels.md \
  --svg-out native/build-scalar/native-distance-kernels.svg

python tools/native_distance_compare.py \
  --avx2-json native/build-avx2/native-distance-kernels.json \
  --scalar-json native/build-scalar/native-distance-kernels.json
```

## Inputs

- AVX2 source: `native/artifacts/avx2/native-distance-kernels.json`
- scalar fallback source: `native/artifacts/scalar/native-distance-kernels.json`
