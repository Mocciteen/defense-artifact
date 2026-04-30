#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/relu_low12_patch.h>
#include <tvm/runtime/tensor.h>

namespace {

using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::runtime::DataType;
using tvm::runtime::Tensor;

struct Args {
  std::string library_path;
  std::string input_bin;
  std::string output_bin;
  std::string output_txt;
  std::string summary_json;
  std::string input_shape;
  std::string entry_name{"main"};
  bool stream_mode{false};
};

void Usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --library <compiled_vm.so>"
            << " --input-shape <1,3,96,96>"
            << " [--input-bin <input_f32.bin>]"
            << " [--output-bin <output_f32.bin>]"
            << " [--output-txt <output.txt>]"
            << " [--summary-json <summary.json>]"
            << " [--stream]"
            << " [--entry <main>]\n";
}

std::string RequireValue(int& i, int argc, char** argv, const char* flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  ++i;
  return argv[i];
}

Args ParseArgs(int argc, char** argv) {
  Args out;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--library") {
      out.library_path = RequireValue(i, argc, argv, "--library");
    } else if (arg == "--input-bin") {
      out.input_bin = RequireValue(i, argc, argv, "--input-bin");
    } else if (arg == "--input-shape") {
      out.input_shape = RequireValue(i, argc, argv, "--input-shape");
    } else if (arg == "--output-bin") {
      out.output_bin = RequireValue(i, argc, argv, "--output-bin");
    } else if (arg == "--output-txt") {
      out.output_txt = RequireValue(i, argc, argv, "--output-txt");
    } else if (arg == "--summary-json") {
      out.summary_json = RequireValue(i, argc, argv, "--summary-json");
    } else if (arg == "--stream") {
      out.stream_mode = true;
    } else if (arg == "--entry") {
      out.entry_name = RequireValue(i, argc, argv, "--entry");
    } else if (arg == "-h" || arg == "--help") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (out.library_path.empty() || out.input_shape.empty()) {
    throw std::runtime_error("missing required arguments");
  }
  if (!out.stream_mode && (out.input_bin.empty() || out.output_bin.empty())) {
    throw std::runtime_error("missing required arguments");
  }
  return out;
}

std::vector<int64_t> ParseShape(const std::string& text) {
  std::vector<int64_t> shape;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      throw std::runtime_error("empty dimension in input shape: " + text);
    }
    char* end = nullptr;
    long long value = std::strtoll(item.c_str(), &end, 10);
    if (end == nullptr || end[0] != '\0' || value <= 0) {
      throw std::runtime_error("invalid dimension in input shape: " + item);
    }
    shape.push_back(static_cast<int64_t>(value));
  }
  if (shape.empty()) {
    throw std::runtime_error("empty input shape");
  }
  return shape;
}

int64_t Product(const std::vector<int64_t>& shape) {
  int64_t product = 1;
  for (int64_t dim : shape) {
    product *= dim;
  }
  return product;
}

std::vector<float> ReadFloat32File(const std::string& path, int64_t expected_elems) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + path);
  }
  in.seekg(0, std::ios::end);
  std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size < 0) {
    throw std::runtime_error("failed to stat input file: " + path);
  }
  const auto expected_bytes = expected_elems * static_cast<int64_t>(sizeof(float));
  if (size != static_cast<std::streamoff>(expected_bytes)) {
    std::ostringstream os;
    os << "unexpected input size for " << path << ": got " << size << " bytes, expected "
       << expected_bytes << " bytes";
    throw std::runtime_error(os.str());
  }
  std::vector<float> data(static_cast<size_t>(expected_elems));
  in.read(reinterpret_cast<char*>(data.data()), size);
  if (!in) {
    throw std::runtime_error("failed to read input file: " + path);
  }
  return data;
}

bool ReadExact(std::istream& in, void* dst, size_t nbytes) {
  char* out = static_cast<char*>(dst);
  size_t total = 0;
  while (total < nbytes) {
    in.read(out + total, static_cast<std::streamsize>(nbytes - total));
    const size_t got = static_cast<size_t>(in.gcount());
    if (got == 0) {
      if (total == 0 && in.eof()) {
        return false;
      }
      throw std::runtime_error("unexpected EOF while reading streamed input");
    }
    total += got;
  }
  return true;
}

void WriteFloat32File(const std::string& path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output bin: " + path);
  }
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write output bin: " + path);
  }
}

void WriteTextFile(const std::string& path, const std::vector<float>& data) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output txt: " + path);
  }
  out << std::setprecision(9);
  for (size_t i = 0; i < data.size(); ++i) {
    out << i << '\t' << data[i] << '\n';
  }
  if (!out) {
    throw std::runtime_error("failed to write output txt: " + path);
  }
}

enum class InputZeroDitherMode {
  kOff,
  kCheckerboard,
  kRandom,
};

enum class InputZeroDitherLayout {
  kNCHW,
  kNHWC,
};

struct InputZeroDitherConfig {
  bool enabled{false};
  InputZeroDitherMode mode{InputZeroDitherMode::kCheckerboard};
  float eps_min{1.0e-8f};
  float eps_max{2.0e-8f};
  float zero_threshold{0.1f};
  InputZeroDitherLayout layout{InputZeroDitherLayout::kNCHW};
  bool has_seed{false};
  uint32_t seed{0u};
  bool silent{false};
};

bool IEquals(const char* lhs, const char* rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return lhs == rhs;
  }
  while (*lhs != '\0' && *rhs != '\0') {
    unsigned char lc = static_cast<unsigned char>(*lhs);
    unsigned char rc = static_cast<unsigned char>(*rhs);
    unsigned char l_lower =
        (lc >= 'A' && lc <= 'Z') ? static_cast<unsigned char>(lc - 'A' + 'a') : lc;
    unsigned char r_lower =
        (rc >= 'A' && rc <= 'Z') ? static_cast<unsigned char>(rc - 'A' + 'a') : rc;
    if (l_lower != r_lower) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

const char* GetEnvWithFallback(const char* primary, const char* fallback) {
  const char* primary_value = std::getenv(primary);
  if (primary_value != nullptr && primary_value[0] != '\0') {
    return primary_value;
  }
  const char* fallback_value = std::getenv(fallback);
  if (fallback_value != nullptr && fallback_value[0] != '\0') {
    return fallback_value;
  }
  return nullptr;
}

bool ParseBoolEnv(const char* primary, const char* fallback) {
  const char* value = GetEnvWithFallback(primary, fallback);
  if (value == nullptr) {
    return false;
  }
  if (IEquals(value, "0") || IEquals(value, "off") || IEquals(value, "false") ||
      IEquals(value, "no")) {
    return false;
  }
  return true;
}

bool ParseFloatEnv(const char* primary, const char* fallback, float* out) {
  const char* value = GetEnvWithFallback(primary, fallback);
  if (value == nullptr) {
    return false;
  }
  char* end = nullptr;
  float parsed = std::strtof(value, &end);
  if (end == nullptr || end[0] != '\0' || !std::isfinite(parsed)) {
    throw std::runtime_error(std::string("invalid input zero dither float env: ") + value);
  }
  *out = parsed;
  return true;
}

bool ParseUint32Env(const char* primary, const char* fallback, uint32_t* out) {
  const char* value = GetEnvWithFallback(primary, fallback);
  if (value == nullptr) {
    return false;
  }
  char* end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 0);
  if (end == nullptr || end[0] != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string("invalid input zero dither uint env: ") + value);
  }
  *out = static_cast<uint32_t>(parsed);
  return true;
}

InputZeroDitherLayout ParseLayoutOrThrow(const char* value) {
  if (IEquals(value, "NCHW")) {
    return InputZeroDitherLayout::kNCHW;
  }
  if (IEquals(value, "NHWC")) {
    return InputZeroDitherLayout::kNHWC;
  }
  throw std::runtime_error(std::string("invalid input zero dither layout: ") + value);
}

InputZeroDitherConfig ParseInputZeroDitherConfig() {
  InputZeroDitherConfig cfg;
  cfg.silent =
      ParseBoolEnv("TVM_INPUT_ZERO_DITHER_SILENT", "GLOW_INPUT_ZERO_DITHER_SILENT");

  const char* mode = GetEnvWithFallback("TVM_INPUT_ZERO_DITHER", "GLOW_INPUT_ZERO_DITHER");
  if (mode == nullptr || IEquals(mode, "0") || IEquals(mode, "off") || IEquals(mode, "none")) {
    return cfg;
  }

  if (IEquals(mode, "checker") || IEquals(mode, "checkerboard")) {
    cfg.mode = InputZeroDitherMode::kCheckerboard;
  } else if (IEquals(mode, "random") || IEquals(mode, "rand")) {
    cfg.mode = InputZeroDitherMode::kRandom;
  } else {
    throw std::runtime_error(std::string("invalid input zero dither mode: ") + mode);
  }
  cfg.enabled = true;
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS0", "GLOW_INPUT_ZERO_DITHER_EPS0", &cfg.eps_min);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS1", "GLOW_INPUT_ZERO_DITHER_EPS1", &cfg.eps_max);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS_MIN", "GLOW_INPUT_ZERO_DITHER_EPS_MIN", &cfg.eps_min);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS_MAX", "GLOW_INPUT_ZERO_DITHER_EPS_MAX", &cfg.eps_max);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_THRESH", "GLOW_INPUT_ZERO_DITHER_THRESH",
                &cfg.zero_threshold);
  cfg.has_seed = ParseUint32Env("TVM_INPUT_ZERO_DITHER_SEED", "GLOW_INPUT_ZERO_DITHER_SEED",
                                &cfg.seed);
  const char* layout =
      GetEnvWithFallback("TVM_INPUT_ZERO_DITHER_LAYOUT", "GLOW_INPUT_ZERO_DITHER_LAYOUT");
  if (layout != nullptr) {
    cfg.layout = ParseLayoutOrThrow(layout);
  }
  return cfg;
}

uint32_t MixHash(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

uint32_t MakeHashSeed(const InputZeroDitherConfig& cfg, size_t linear_index, size_t y, size_t x) {
  uint64_t index64 = static_cast<uint64_t>(linear_index);
  uint32_t seed = cfg.has_seed ? cfg.seed : 0x6d2b79f5U;
  seed ^= static_cast<uint32_t>(index64);
  seed ^= static_cast<uint32_t>(index64 >> 32) * 0x9e3779b9U;
  seed ^= static_cast<uint32_t>(y) * 0x85ebca6bU;
  seed ^= static_cast<uint32_t>(x) * 0xc2b2ae35U;
  return MixHash(seed);
}

float ComputeDelta(const InputZeroDitherConfig& cfg, size_t linear_index, size_t y, size_t x) {
  if (cfg.mode == InputZeroDitherMode::kCheckerboard) {
    return ((y + x) & 1U) == 0 ? cfg.eps_min : cfg.eps_max;
  }
  uint32_t hash = MakeHashSeed(cfg, linear_index, y, x);
  float unit = static_cast<float>(hash >> 8) * (1.0f / 16777216.0f);
  return cfg.eps_min + (cfg.eps_max - cfg.eps_min) * unit;
}

void LogInputZeroDitherSummary(const InputZeroDitherConfig& cfg, size_t changed, size_t total) {
  if (cfg.silent || !cfg.enabled) {
    return;
  }
  std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10);
  if (cfg.mode == InputZeroDitherMode::kCheckerboard) {
    std::cerr << "input_zero_dither=checkerboard";
  } else {
    std::cerr << "input_zero_dither=random";
  }
  std::cerr << " layout=" << (cfg.layout == InputZeroDitherLayout::kNCHW ? "NCHW" : "NHWC")
            << " eps_min=" << cfg.eps_min << " eps_max=" << cfg.eps_max
            << " thresh=" << cfg.zero_threshold << " changed=" << changed << "/" << total;
  if (cfg.has_seed) {
    std::cerr << " seed=" << cfg.seed;
  } else {
    std::cerr << " seed=default_hash";
  }
  std::cerr << "\n";
}

struct ScopedInputDitherDisable {
  std::vector<std::pair<std::string, std::optional<std::string>>> saved;

  ScopedInputDitherDisable() {
    static const char* kKeys[] = {
        "TVM_INPUT_ZERO_DITHER",        "TVM_INPUT_ZERO_DITHER_LAYOUT",
        "TVM_INPUT_ZERO_DITHER_THRESH", "TVM_INPUT_ZERO_DITHER_EPS_MIN",
        "TVM_INPUT_ZERO_DITHER_EPS_MAX", "TVM_INPUT_ZERO_DITHER_EPS0",
        "TVM_INPUT_ZERO_DITHER_EPS1",   "TVM_INPUT_ZERO_DITHER_SEED",
        "TVM_INPUT_ZERO_DITHER_SILENT", "GLOW_INPUT_ZERO_DITHER",
        "GLOW_INPUT_ZERO_DITHER_LAYOUT", "GLOW_INPUT_ZERO_DITHER_THRESH",
        "GLOW_INPUT_ZERO_DITHER_EPS_MIN", "GLOW_INPUT_ZERO_DITHER_EPS_MAX",
        "GLOW_INPUT_ZERO_DITHER_EPS0",  "GLOW_INPUT_ZERO_DITHER_EPS1",
        "GLOW_INPUT_ZERO_DITHER_SEED",  "GLOW_INPUT_ZERO_DITHER_SILENT",
    };
    for (const char* key : kKeys) {
      const char* value = std::getenv(key);
      if (value != nullptr) {
        saved.emplace_back(key, std::string(value));
        unsetenv(key);
      } else {
        saved.emplace_back(key, std::nullopt);
      }
    }
  }

  ~ScopedInputDitherDisable() {
    for (const auto& [key, value] : saved) {
      if (value.has_value()) {
        setenv(key.c_str(), value->c_str(), 1);
      } else {
        unsetenv(key.c_str());
      }
    }
  }
};

std::vector<int> TopKIndices(const std::vector<float>& data, size_t k) {
  std::vector<int> indices(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    indices[i] = static_cast<int>(i);
  }
  if (k > indices.size()) {
    k = indices.size();
  }
  std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                    indices.end(), [&data](int lhs, int rhs) {
                      return data[static_cast<size_t>(lhs)] >
                             data[static_cast<size_t>(rhs)];
                    });
  indices.resize(k);
  return indices;
}

Function RequireFunction(const Module& mod, const std::string& name) {
  auto opt = mod->GetFunction(name, false);
  if (!opt.has_value()) {
    throw std::runtime_error("missing module function: " + name);
  }
  return *opt;
}

Tensor AllocateInputTensor(const std::vector<int64_t>& shape) {
  return Tensor::Empty(tvm::ffi::Shape(shape), DataType(DataType::kFloat, 32, 1),
                       DLDevice{kDLCPU, 0});
}

void FillInputTensorFromBytes(Tensor& tensor, const float* data, int64_t expected_elems,
                              const std::vector<int64_t>& shape,
                              const InputZeroDitherConfig& dither_cfg) {
  const size_t nbytes = static_cast<size_t>(expected_elems) * sizeof(float);
  if (!dither_cfg.enabled || shape.size() != 4) {
    tensor.CopyFromBytes(data, nbytes);
    return;
  }

  DLTensor* raw_tensor = const_cast<DLTensor*>(tensor.operator->());
  float* dst = static_cast<float*>(raw_tensor->data);
  int64_t n = shape[0];
  int64_t c = dither_cfg.layout == InputZeroDitherLayout::kNCHW ? shape[1] : shape[3];
  int64_t h = dither_cfg.layout == InputZeroDitherLayout::kNCHW ? shape[2] : shape[1];
  int64_t w = dither_cfg.layout == InputZeroDitherLayout::kNCHW ? shape[3] : shape[2];
  size_t changed = 0;

  if (dither_cfg.layout == InputZeroDitherLayout::kNCHW) {
    for (int64_t ni = 0; ni < n; ++ni) {
      for (int64_t ci = 0; ci < c; ++ci) {
        for (int64_t yi = 0; yi < h; ++yi) {
          for (int64_t xi = 0; xi < w; ++xi) {
            size_t index = static_cast<size_t>(((ni * c + ci) * h + yi) * w + xi);
            float value = data[index];
            if (std::fabs(value) <= dither_cfg.zero_threshold) {
              value += ComputeDelta(dither_cfg, index, static_cast<size_t>(yi),
                                    static_cast<size_t>(xi));
              ++changed;
            }
            dst[index] = value;
          }
        }
      }
    }
  } else {
    for (int64_t ni = 0; ni < n; ++ni) {
      for (int64_t yi = 0; yi < h; ++yi) {
        for (int64_t xi = 0; xi < w; ++xi) {
          for (int64_t ci = 0; ci < c; ++ci) {
            size_t index = static_cast<size_t>(((ni * h + yi) * w + xi) * c + ci);
            float value = data[index];
            if (std::fabs(value) <= dither_cfg.zero_threshold) {
              value += ComputeDelta(dither_cfg, index, static_cast<size_t>(yi),
                                    static_cast<size_t>(xi));
              ++changed;
            }
            dst[index] = value;
          }
        }
      }
    }
  }
  LogInputZeroDitherSummary(dither_cfg, changed, static_cast<size_t>(expected_elems));
}

std::string ShapeToJson(const tvm::ffi::ShapeView& shape) {
  std::ostringstream os;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << shape[i];
  }
  return os.str();
}

void WriteSummaryJson(const std::string& path, const Args& args,
                      const std::vector<int64_t>& input_shape, double tvm_seconds,
                      const tvm::ffi::ShapeView& output_shape,
                      const std::vector<float>& output) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open summary json: " + path);
  }
  const auto top3 = TopKIndices(output, 3);
  const int pred =
      static_cast<int>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));

  out << "{\n";
  out << "  \"entry_name\": " << std::quoted(args.entry_name) << ",\n";
  out << "  \"input_bin\": " << std::quoted(args.input_bin) << ",\n";
  out << "  \"library_path\": " << std::quoted(args.library_path) << ",\n";
  out << "  \"input_shape\": [";
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << input_shape[i];
  }
  out << "],\n";
  out << "  \"tvm_seconds\": " << std::fixed << std::setprecision(6) << tvm_seconds << ",\n";
  out << "  \"tvm\": {\n";
  out << "    \"name\": \"tvm_native_vm\",\n";
  out << "    \"shape\": [" << ShapeToJson(output_shape) << "],\n";
  out << "    \"pred\": " << pred << ",\n";
  out << "    \"top3_indices\": [";
  for (size_t i = 0; i < top3.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << top3[i];
  }
  out << "],\n";
  out << "    \"top3_logits\": [";
  for (size_t i = 0; i < top3.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << output[static_cast<size_t>(top3[i])];
  }
  out << "],\n";
  float sum = 0.0f;
  float min_v = output.front();
  float max_v = output.front();
  for (float value : output) {
    sum += value;
    min_v = std::min(min_v, value);
    max_v = std::max(max_v, value);
  }
  out << "    \"logits_sum\": " << sum << ",\n";
  out << "    \"logits_min\": " << min_v << ",\n";
  out << "    \"logits_max\": " << max_v << "\n";
  out << "  }\n";
  out << "}\n";
  if (!out) {
    throw std::runtime_error("failed to write summary json: " + path);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const std::vector<int64_t> input_shape = ParseShape(args.input_shape);
    const int64_t input_elems = Product(input_shape);
    const InputZeroDitherConfig dither_cfg = ParseInputZeroDitherConfig();
    ScopedInputDitherDisable disable_runtime_dither;

    Module exec = Module::LoadFromFile(args.library_path);
    Function vm_load_executable = RequireFunction(exec, "vm_load_executable");
    Module vm = vm_load_executable().cast<Module>();

    RequireFunction(vm, "vm_initialization")(static_cast<int>(kDLCPU), 0, 2);

    Tensor input_tensor = AllocateInputTensor(input_shape);

    Function set_input = RequireFunction(vm, "set_input");
    Function invoke_stateful = RequireFunction(vm, "invoke_stateful");
    Function get_output_arity = RequireFunction(vm, "get_output_arity");
    Function get_output = RequireFunction(vm, "get_output");

    if (args.stream_mode) {
      std::vector<float> input(static_cast<size_t>(input_elems));
      while (ReadExact(std::cin, input.data(), input.size() * sizeof(float))) {
        FillInputTensorFromBytes(input_tensor, input.data(), input_elems, input_shape, dither_cfg);
        set_input(args.entry_name, input_tensor);
        tvm::runtime::relulow12::ResetInferencePatchSeed();
        invoke_stateful(args.entry_name);

        const int output_arity = get_output_arity(args.entry_name).cast<int>();
        if (output_arity != -1 && output_arity != 1) {
          std::ostringstream os;
          os << "unexpected output arity: " << output_arity;
          throw std::runtime_error(os.str());
        }

        Tensor output_tensor =
            (output_arity == -1) ? get_output(args.entry_name).cast<Tensor>()
                                 : get_output(args.entry_name, 0).cast<Tensor>();
        const int64_t output_elems = output_tensor.Shape().Product();
        std::vector<float> output(static_cast<size_t>(output_elems));
        output_tensor.CopyToBytes(output.data(), output.size() * sizeof(float));
        std::cout.write(reinterpret_cast<const char*>(output.data()),
                        static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!std::cout) {
          throw std::runtime_error("failed to write streamed output");
        }
        std::cout.flush();
      }
      return 0;
    }

    const std::vector<float> input = ReadFloat32File(args.input_bin, input_elems);
    FillInputTensorFromBytes(input_tensor, input.data(), input_elems, input_shape, dither_cfg);
    set_input(args.entry_name, input_tensor);
    const auto start = std::chrono::steady_clock::now();
    tvm::runtime::relulow12::ResetInferencePatchSeed();
    invoke_stateful(args.entry_name);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    const int output_arity = get_output_arity(args.entry_name).cast<int>();
    if (output_arity != -1 && output_arity != 1) {
      std::ostringstream os;
      os << "unexpected output arity: " << output_arity;
      throw std::runtime_error(os.str());
    }

    Tensor output_tensor =
        (output_arity == -1) ? get_output(args.entry_name).cast<Tensor>()
                             : get_output(args.entry_name, 0).cast<Tensor>();
    const auto output_shape = output_tensor.Shape();
    const int64_t output_elems = output_shape.Product();
    std::vector<float> output(static_cast<size_t>(output_elems));
    output_tensor.CopyToBytes(output.data(), output.size() * sizeof(float));

    WriteFloat32File(args.output_bin, output);
    if (!args.output_txt.empty()) {
      WriteTextFile(args.output_txt, output);
    }
    if (!args.summary_json.empty()) {
      WriteSummaryJson(args.summary_json, args, input_shape, elapsed.count(), output_shape, output);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 1;
  }
}
