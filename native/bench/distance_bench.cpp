#include "vecadvisor/distance.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kCorpusSalt = 0xA511E9B3ULL;
constexpr std::uint64_t kQuerySalt = 0x63D83595ULL;

struct Options {
  std::size_t rows = 4096;
  std::size_t queries = 16;
  std::size_t dim = 128;
  std::size_t iterations = 5;
  std::vector<std::string> metrics{"l2", "ip", "cosine"};
  bool json = false;
};

struct Timing {
  double seconds;
  double checksum;
  std::uint64_t distances;
};

struct MetricSpec {
  const char* name;
  float (*scalar_fn)(const float*, const float*, std::size_t) noexcept;
  float (*dispatch_fn)(const float*, const float*, std::size_t) noexcept;
  const char* selected_kernel;
};

std::uint32_t mixed_value(std::size_t primary, std::size_t secondary, std::uint64_t salt) noexcept {
  std::uint64_t value =
      (static_cast<std::uint64_t>(primary + 1U) * 2654435761ULL) ^
      (static_cast<std::uint64_t>(secondary + 1U) * 2246822519ULL) ^
      (salt * 3266489917ULL);
  value ^= value >> 16U;
  value *= 2246822519ULL;
  value ^= value >> 13U;
  return static_cast<std::uint32_t>(value);
}

float generated_value(std::size_t row, std::size_t dim_index, std::uint64_t salt) noexcept {
  const std::uint32_t low_bits = mixed_value(row, dim_index, salt) & 0xFFFFU;
  return static_cast<float>(low_bits) / 32768.0F - 1.0F;
}

std::vector<float> make_matrix(std::size_t rows, std::size_t dim, std::uint64_t salt) {
  std::vector<float> values(rows * dim);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      values[row * dim + col] = generated_value(row, col, salt);
    }
  }
  return values;
}

std::size_t parse_size(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  std::size_t consumed = 0;
  try {
    parsed = std::stoull(std::string(value), &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid integer for " + std::string(label));
  }
  if (consumed != value.size() || parsed == 0) {
    throw std::invalid_argument("invalid positive integer for " + std::string(label));
  }
  return parsed;
}

std::vector<std::string> split_metrics(std::string_view text) {
  std::vector<std::string> metrics;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    if (end > start) {
      metrics.emplace_back(text.substr(start, end - start));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  if (metrics.empty()) {
    throw std::invalid_argument("--metrics must include at least one metric");
  }
  for (const auto& metric : metrics) {
    if (metric != "l2" && metric != "ip" && metric != "cosine") {
      throw std::invalid_argument("unsupported metric: " + metric);
    }
  }
  return metrics;
}

void print_help(const char* program) {
  std::cout << "Usage: " << program
            << " [--rows N] [--queries N] [--dim N] [--iterations N]"
               " [--metrics l2,ip,cosine] [--json]\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    auto require_value = [&](std::string_view label) -> std::string_view {
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + std::string(label));
      }
      ++index;
      return argv[index];
    };

    if (arg == "--rows") {
      options.rows = parse_size(require_value(arg), arg);
    } else if (arg == "--queries") {
      options.queries = parse_size(require_value(arg), arg);
    } else if (arg == "--dim") {
      options.dim = parse_size(require_value(arg), arg);
    } else if (arg == "--iterations") {
      options.iterations = parse_size(require_value(arg), arg);
    } else if (arg == "--metrics") {
      options.metrics = split_metrics(require_value(arg));
    } else if (arg == "--json") {
      options.json = true;
    } else if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
  }
  return options;
}

Timing run_kernel(
    float (*kernel)(const float*, const float*, std::size_t) noexcept,
    const std::vector<float>& corpus,
    const std::vector<float>& query_vectors,
    std::size_t rows,
    std::size_t queries,
    std::size_t dim,
    std::size_t iterations) {
  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    for (std::size_t query = 0; query < queries; ++query) {
      const float* query_vector = query_vectors.data() + query * dim;
      for (std::size_t row = 0; row < rows; ++row) {
        checksum += kernel(query_vector, corpus.data() + row * dim, dim);
      }
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration<double>(end - start).count();
  return Timing{elapsed, checksum, static_cast<std::uint64_t>(iterations * queries * rows)};
}

double max_abs_error(
    float (*left_kernel)(const float*, const float*, std::size_t) noexcept,
    float (*right_kernel)(const float*, const float*, std::size_t) noexcept,
    const std::vector<float>& corpus,
    const std::vector<float>& query_vectors,
    std::size_t rows,
    std::size_t queries,
    std::size_t dim) {
  const std::size_t checked_rows = std::min<std::size_t>(rows, 256);
  const std::size_t checked_queries = std::min<std::size_t>(queries, 8);
  double max_error = 0.0;
  for (std::size_t query = 0; query < checked_queries; ++query) {
    const float* query_vector = query_vectors.data() + query * dim;
    for (std::size_t row = 0; row < checked_rows; ++row) {
      const float left = left_kernel(query_vector, corpus.data() + row * dim, dim);
      const float right = right_kernel(query_vector, corpus.data() + row * dim, dim);
      max_error = std::max(max_error, std::fabs(static_cast<double>(left) - right));
    }
  }
  return max_error;
}

double ns_per_distance(const Timing& timing) {
  if (timing.distances == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return timing.seconds * 1.0e9 / static_cast<double>(timing.distances);
}

double distances_per_second(const Timing& timing) {
  if (timing.seconds == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(timing.distances) / timing.seconds;
}

std::vector<MetricSpec> metric_specs(const vecadvisor::native::KernelCapabilities& capabilities) {
  return {
      MetricSpec{
          "l2",
          vecadvisor::native::l2_squared_scalar,
          vecadvisor::native::l2_squared,
          capabilities.l2_kernel,
      },
      MetricSpec{
          "ip",
          vecadvisor::native::inner_product_scalar,
          vecadvisor::native::inner_product,
          capabilities.inner_product_kernel,
      },
      MetricSpec{
          "cosine",
          vecadvisor::native::cosine_distance_scalar,
          vecadvisor::native::cosine_distance,
          capabilities.cosine_kernel,
      },
  };
}

const MetricSpec& find_metric(
    const std::vector<MetricSpec>& specs,
    const std::string& metric_name) {
  const auto iter = std::find_if(specs.begin(), specs.end(), [&](const MetricSpec& spec) {
    return metric_name == spec.name;
  });
  if (iter == specs.end()) {
    throw std::invalid_argument("unsupported metric: " + metric_name);
  }
  return *iter;
}

void print_timing_json(const char* name, const Timing& timing) {
  std::cout << '"' << name << "\":{"
            << "\"seconds\":" << timing.seconds << ','
            << "\"checksum\":" << timing.checksum << ','
            << "\"distances\":" << timing.distances << ','
            << "\"ns_per_distance\":" << ns_per_distance(timing) << ','
            << "\"distances_per_second\":" << distances_per_second(timing) << '}';
}

void print_json_result(
    const Options& options,
    const vecadvisor::native::KernelCapabilities& capabilities,
    const std::vector<float>& corpus,
    const std::vector<float>& query_vectors) {
  const auto specs = metric_specs(capabilities);
  std::cout << std::setprecision(12);
  std::cout << "{"
            << "\"schema_version\":1,"
            << "\"rows\":" << options.rows << ','
            << "\"queries\":" << options.queries << ','
            << "\"dim\":" << options.dim << ','
            << "\"iterations\":" << options.iterations << ','
            << "\"data_generator\":\"vecadvisor_native_v1\","
            << "\"capabilities\":{"
            << "\"avx2_compiled\":" << (capabilities.avx2_compiled ? "true" : "false") << ','
            << "\"avx2_runtime\":" << (capabilities.avx2_runtime ? "true" : "false")
            << "},"
            << "\"results\":[";

  bool first = true;
  for (const auto& metric_name : options.metrics) {
    const auto& spec = find_metric(specs, metric_name);
    const auto scalar =
        run_kernel(spec.scalar_fn, corpus, query_vectors, options.rows, options.queries, options.dim, options.iterations);
    const auto dispatch =
        run_kernel(spec.dispatch_fn, corpus, query_vectors, options.rows, options.queries, options.dim, options.iterations);
    const double error =
        max_abs_error(spec.scalar_fn, spec.dispatch_fn, corpus, query_vectors, options.rows, options.queries, options.dim);
    const double scalar_ns = ns_per_distance(scalar);
    const double dispatch_ns = ns_per_distance(dispatch);
    const double speedup = dispatch_ns == 0.0 ? std::numeric_limits<double>::infinity()
                                              : scalar_ns / dispatch_ns;

    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << "{"
              << "\"metric\":\"" << spec.name << "\","
              << "\"selected_kernel\":\"" << spec.selected_kernel << "\",";
    print_timing_json("scalar", scalar);
    std::cout << ',';
    print_timing_json("dispatch", dispatch);
    std::cout << ",\"speedup_vs_scalar\":" << speedup
              << ",\"max_abs_error_vs_scalar\":" << error << "}";
  }
  std::cout << "]}\n";
}

void print_text_result(
    const Options& options,
    const vecadvisor::native::KernelCapabilities& capabilities,
    const std::vector<float>& corpus,
    const std::vector<float>& query_vectors) {
  const auto specs = metric_specs(capabilities);
  std::cout << "VecAdvisor native distance benchmark\n"
            << "rows=" << options.rows << " queries=" << options.queries
            << " dim=" << options.dim << " iterations=" << options.iterations
            << " avx2_compiled=" << capabilities.avx2_compiled
            << " avx2_runtime=" << capabilities.avx2_runtime << "\n\n";
  std::cout << std::left << std::setw(10) << "metric" << std::setw(12) << "kernel"
            << std::right << std::setw(16) << "scalar ns" << std::setw(16) << "dispatch ns"
            << std::setw(12) << "speedup" << std::setw(16) << "max error" << '\n';

  for (const auto& metric_name : options.metrics) {
    const auto& spec = find_metric(specs, metric_name);
    const auto scalar =
        run_kernel(spec.scalar_fn, corpus, query_vectors, options.rows, options.queries, options.dim, options.iterations);
    const auto dispatch =
        run_kernel(spec.dispatch_fn, corpus, query_vectors, options.rows, options.queries, options.dim, options.iterations);
    const double error =
        max_abs_error(spec.scalar_fn, spec.dispatch_fn, corpus, query_vectors, options.rows, options.queries, options.dim);
    const double scalar_ns = ns_per_distance(scalar);
    const double dispatch_ns = ns_per_distance(dispatch);
    const double speedup = dispatch_ns == 0.0 ? std::numeric_limits<double>::infinity()
                                              : scalar_ns / dispatch_ns;
    std::cout << std::left << std::setw(10) << spec.name << std::setw(12) << spec.selected_kernel
              << std::right << std::setw(16) << scalar_ns << std::setw(16) << dispatch_ns
              << std::setw(12) << speedup << std::setw(16) << error << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    const auto corpus = make_matrix(options.rows, options.dim, kCorpusSalt);
    const auto query_vectors = make_matrix(options.queries, options.dim, kQuerySalt);
    const auto capabilities = vecadvisor::native::detect_capabilities();

    if (options.json) {
      print_json_result(options, capabilities, corpus, query_vectors);
    } else {
      print_text_result(options, capabilities, corpus, query_vectors);
    }
  } catch (const std::exception& exc) {
    std::cerr << "vecadvisor_distance_bench: " << exc.what() << '\n';
    return 2;
  }
  return 0;
}
