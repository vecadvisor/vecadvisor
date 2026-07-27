# Native Distance Kernels

VecAdvisor MVP2 starts with native exact-distance kernels. The first native
target is intentionally small: a C++17 static library with scalar kernels and
optional AVX2 implementations for:

- squared L2 distance,
- inner product,
- cosine distance.

The Python package still uses the existing NumPy ground-truth path. Native code
is not wired into PyPI packaging yet. This keeps the published CLI stable while
the native kernel ABI, benchmark harness, and cross-platform build discipline
settle.

## Build

Install a C++17 compiler and CMake, then run:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release --parallel
ctest --test-dir native/build --output-on-failure
```

AVX2 is enabled automatically on x86/x86_64 when the compiler supports the
required flag. Runtime dispatch still checks CPU support before calling AVX2
kernels.

To force scalar-only kernels:

```bash
cmake -S native -B native/build -DVECADVISOR_NATIVE_ENABLE_AVX2=OFF
```

## Design Rules

- Keep exact search blocked; never materialize an `N x Q` distance matrix.
- Keep scalar kernels as the correctness baseline for every SIMD path.
- Add SIMD by metric and instruction set behind runtime dispatch.
- Do not change the Python package build until native ABI and CI are stable.
- Prefer reproducible native benchmarks before claiming speedups in README or
  launch material.
