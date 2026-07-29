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

## Data Contract

The current native API is deliberately narrow and easy to bind later:

- inputs are contiguous `float32` arrays;
- vectors are row-major when passed as a matrix by a caller;
- kernels accept unaligned pointers and use unaligned SIMD loads;
- `dim` may be any positive value and does not need to be divisible by the SIMD
  lane width;
- callers must pass non-null pointers when `dim > 0`;
- scalar kernels are the correctness baseline for every dispatch path.

The kernels do not sanitize `NaN` or infinite values on the hot path. IEEE
floating-point behavior is preserved: `NaN` inputs propagate to the result, and
infinite values follow the platform math rules. Cosine distance clamps a zero
or near-zero denominator to `1e-12` to avoid division by zero, matching the
project's existing Python benchmark semantics.

SIMD reductions can differ slightly from scalar reduction order. Tests and
benchmarks should treat dispatch results within `1e-4` absolute error of the
scalar path as equivalent for normal benchmark vectors.

## ABI And Python Binding Strategy

The C++ namespace API in `vecadvisor/distance.hpp` is for native callers. The
stable boundary for language bindings is the C ABI in
`vecadvisor/distance_c.h`. That header exposes:

- `vecadvisor_distance_compute` for one distance between two vectors;
- `vecadvisor_distance_compute_many` for one query vector against a row-major
  corpus matrix;
- `vecadvisor_distance_get_capabilities` for runtime dispatch visibility;
- status codes instead of exceptions.

Python integration should bind to the C ABI, not to C++ symbols. The planned
adapter is an optional Python module that loads the shared library and exposes
NumPy-array entry points for exact ground truth and candidate rescoring. The
advisor, cost model, SQL parsing, and recommendation logic remain pure Python.
If the shared library is missing or rejects an input, Python falls back to the
existing NumPy implementation.

The initial binding should cross the native boundary once per query chunk using
`vecadvisor_distance_compute_many`; it should not call
`vecadvisor_distance_compute` once per row. That keeps Python overhead out of
the hot loop and preserves the SIMD speedup shown in the benchmark artifact.

The C ABI is versioned conservatively by header shape: append new enum values
and functions, but do not reorder existing enum values or fields in
`vecadvisor_kernel_capabilities`.

CMake builds both targets:

- `vecadvisor_distance`, a static library used by native tests and tools;
- `vecadvisor_distance_shared`, a shared library with C ABI exports for future
  Python loading.

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

## Benchmark

The native benchmark executable is built as `vecadvisor_distance_bench` when
`VECADVISOR_NATIVE_BUILD_BENCHMARKS=ON`:

```bash
native/build/vecadvisor_distance_bench \
  --rows 4096 \
  --queries 16 \
  --dim 128 \
  --iterations 5 \
  --metrics l2,ip,cosine
```

For a Python-facing report with NumPy checksum validation:

```bash
python tools/native_distance_benchmark.py \
  --rows 4096 \
  --queries 16 \
  --dim 128 \
  --iterations 5
```

The wrapper builds the native target by default, runs the benchmark, validates
scalar and dispatch checksums against deterministic NumPy data, and writes:

- `docs/benchmarks/native-distance-kernels.json`
- `docs/benchmarks/native-distance-kernels.md`
- `docs/assets/native-distance-kernels.svg`

Use `--no-build` when the CMake build directory already exists.

To build the scalar fallback path through the wrapper:

```bash
python tools/native_distance_benchmark.py \
  --build-dir native/build-scalar \
  --disable-avx2
```

The committed MVP2 benchmark artifact compares an AVX2-enabled build with a
scalar-only fallback build:

- `docs/benchmarks/native-distance-kernels.md`
- `docs/benchmarks/native-distance-kernels.json`
- `docs/assets/native-distance-kernels.svg`

Regenerate the combined artifact with:

```bash
python tools/native_distance_compare.py \
  --avx2-json native/build-avx2/native-distance-kernels.json \
  --scalar-json native/build-scalar/native-distance-kernels.json
```

## Design Rules

- Keep exact search blocked; never materialize an `N x Q` distance matrix.
- Keep scalar kernels as the correctness baseline for every SIMD path.
- Add SIMD by metric and instruction set behind runtime dispatch.
- Do not change the Python package build until native ABI and CI are stable.
- Prefer reproducible native benchmarks before claiming speedups in README or
  launch material.
