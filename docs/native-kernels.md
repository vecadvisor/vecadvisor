# Native Distance Kernels

VecAdvisor MVP2 starts with native exact-distance kernels. The first native
target is intentionally small: a C++17 static library with scalar kernels and
optional AVX2 implementations for:

- squared L2 distance,
- inner product,
- cosine distance.

The Python package does not bundle the native shared library yet. When
`VECADVISOR_NATIVE_DISTANCE_LIB` points at a locally built shared library,
Python exact ground-truth can use the native bounded top-k path. When the
library is absent or rejects an input, VecAdvisor falls back to the existing
NumPy implementation. This keeps the published CLI stable while the native
kernel ABI, benchmark harness, and cross-platform build discipline settle.

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

The AVX2 path is compiled only for `distance_avx2.cpp` and uses FMA plus four
independent accumulators in the main loop. That keeps the rest of the binary
portable while reducing the add/multiply dependency chain inside the hot
kernel. Runtime dispatch checks both OS vector-state support and the required
CPU feature bits before calling that code.

## ABI And Python Binding Strategy

The C++ namespace API in `vecadvisor/distance.hpp` is for native callers. The
stable boundary for language bindings is the C ABI in
`vecadvisor/distance_c.h`. That header exposes:

- `vecadvisor_distance_compute` for one distance between two vectors;
- `vecadvisor_distance_compute_many` for one query vector against a row-major
  corpus matrix;
- `vecadvisor_distance_topk` for one query vector against a row-major corpus
  matrix, returning the best row offsets and metric values without
  materializing all distances;
- `vecadvisor_distance_get_capabilities` for runtime dispatch visibility;
- status codes instead of exceptions.

Python integration binds to the C ABI, not to C++ symbols. The optional
`ctypes` adapter loads the shared library and exposes NumPy-array entry points
for exact ground truth. The advisor, cost model, SQL parsing, and
recommendation logic remain pure Python.

The Python ground-truth path crosses the native boundary once per query block
using `vecadvisor_distance_topk`; it does not call
`vecadvisor_distance_compute` once per row. That keeps Python overhead out of
the hot loop and preserves the SIMD speedup shown in the benchmark artifact.
For cosine batches, `vecadvisor_distance_compute_many` caches the query norm
once per call and reuses it for every corpus row.
`vecadvisor_distance_topk` uses the same metric kernels, keeps only `k`
candidates plus the current input block in memory, and returns row offsets
relative to the supplied block. For L2 and cosine, smaller distances rank
first. For inner product, larger scores rank first and the returned
`out_distances` values are the raw inner products. Ties are stable by smaller
row offset.

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
cmake -S native -B native/build
cmake --build native/build --config Release --parallel
ctest --test-dir native/build --output-on-failure
```

Single-config generators default to `Release` when `CMAKE_BUILD_TYPE` is not
provided. Passing `-DCMAKE_BUILD_TYPE=Release` remains fine and is what CI uses
for explicitness.

AVX2 is enabled automatically on x86/x86_64 when the compiler supports the
required AVX2 and FMA flags. Runtime dispatch still checks CPU support before
calling AVX2/FMA kernels.

To force scalar-only kernels:

```bash
cmake -S native -B native/build -DVECADVISOR_NATIVE_ENABLE_AVX2=OFF
```

On Windows, install CMake and Visual Studio Build Tools with the C++ workload,
then run the helper from PowerShell:

```powershell
.\tools\windows_native_build.ps1
```

The helper locates `vcvars64.bat`, configures a Visual Studio x64 build in
`native/build-msvc`, runs native CTest, and then runs the Python native-wrapper
smoke test against the produced `vecadvisor_distance.dll`.

## Use From Python

After building the shared library, point the Python process at it:

```powershell
$env:VECADVISOR_NATIVE_DISTANCE_LIB = "C:\path\to\pgVector\native\build\Release\vecadvisor_distance.dll"
```

```bash
export VECADVISOR_NATIVE_DISTANCE_LIB="$PWD/native/build/libvecadvisor_distance.so"
```

On macOS, use the corresponding `.dylib` path. Benchmark JSON includes
`ground_truth.native_used` so runs can distinguish native-backed exact
ground-truth from the NumPy fallback. The library is optional; no runtime
failure should occur simply because it is not installed.

Native CI builds the shared library and runs
`tests/test_native_distance_integration.py` with
`VECADVISOR_NATIVE_DISTANCE_LIB` set, so the C ABI, Python `ctypes` wrapper,
and blocked exact ground-truth path are validated together.

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

Add an optional SimSIMD baseline when the `simsimd` Python package is
installed:

```bash
python tools/native_distance_benchmark.py \
  --rows 4096 \
  --queries 16 \
  --dim 128 \
  --iterations 5 \
  --external-baselines simsimd
```

SimSIMD is used only by the benchmark wrapper. It is not a VecAdvisor runtime
dependency and is not required to build or use the CLI. The committed Native CI
artifact runs this optional baseline so the public report compares VecAdvisor
AVX2 dispatch against both the local scalar fallback and an external SIMD
library. hnswlib is deferred for now because its primary value is ANN index
behavior rather than a direct drop-in distance-kernel baseline; it fits better
with future native exact-ground-truth and ANN comparison work.

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
- Use bounded top-k selection for exact ground truth instead of sorting a full
  corpus distance vector when only `k` neighbors are needed.
- Keep scalar kernels as the correctness baseline for every SIMD path.
- Add SIMD by metric and instruction set behind runtime dispatch.
- Do not bundle native binaries in the Python package until native ABI and CI
  are stable across platforms.
- Prefer reproducible native benchmarks before claiming speedups in README or
  launch material.

## Roadmap

The native layer is not complete yet. The next credibility items are:

- Package the shared native library in platform wheels once the ABI and CI
  matrix are stable.
- Add an ARM NEON path so Apple Silicon and ARM server users do not see only
  the scalar implementation.
- Add AVX-512 after NEON if the benchmark evidence justifies the extra code.
- Add fp16 and int8 kernels once the float32 path has stable bindings and
  external baselines.
- Expand the Rust/pgrx extension scaffold into read-only SQL advisor functions
  once Python parity fixtures are in place.
