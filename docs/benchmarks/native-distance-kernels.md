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

| metric | AVX2 kernel | AVX2 ns/dist | scalar fallback ns/dist | speedup | max AVX2 error | NumPy checks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| l2 | avx2 | 11.472 | 84.041 | 7.326x | 5.34058e-05 | pass |
| ip | avx2 | 11.291 | 82.212 | 7.281x | 4.76837e-06 | pass |
| cosine | avx2 | 21.011 | 132.897 | 6.325x | 1.19209e-07 | pass |

## Interpretation

The `AVX2 ns/dist` column is the runtime-dispatch path future Python bindings would call on an AVX2-capable host. The scalar fallback column comes from a separate build configured with `-DVECADVISOR_NATIVE_ENABLE_AVX2=OFF`, so it verifies the portable path used on machines without AVX2 support.

## Reproduce

```bash
python tools/native_distance_benchmark.py \
  --build-dir native/build-avx2 \
  --rows 2048 \
  --queries 8 \
  --dim 128 \
  --iterations 5 \
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
