/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file input_zero_dither.cc
 * \brief Input zero dither helpers used at input materialization time.
 */

#include <tvm/runtime/input_zero_dither.h>

#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>

namespace tvm {
namespace runtime {
namespace inputzerodither {
namespace {

enum class InputZeroDitherMode {
  kCheckerboard,
  kRandom,
};

enum class InputZeroDitherLayout {
  kNCHW,
  kNHWC,
};

struct RuntimeConfig {
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

RuntimeConfig g_runtime_config;
std::once_flag g_runtime_config_init_flag;

inline bool IEquals(const char* lhs, const char* rhs) {
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

inline const char* GetEnvWithFallback(const char* primary, const char* fallback) {
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
  TVM_FFI_CHECK(end != nullptr && end[0] == '\0' && std::isfinite(parsed), ValueError)
      << "Invalid input zero dither float env: " << value;
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
  TVM_FFI_CHECK(end != nullptr && end[0] == '\0' &&
                    parsed <= std::numeric_limits<uint32_t>::max(),
                ValueError)
      << "Invalid input zero dither uint env: " << value;
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
  TVM_FFI_THROW(ValueError) << "Invalid input zero dither layout: " << value;
}

void InitializeRuntimeConfig() {
  RuntimeConfig cfg;
  cfg.silent =
      ParseBoolEnv("TVM_INPUT_ZERO_DITHER_SILENT", "GLOW_INPUT_ZERO_DITHER_SILENT");

  const char* mode = GetEnvWithFallback("TVM_INPUT_ZERO_DITHER", "GLOW_INPUT_ZERO_DITHER");
  if (mode == nullptr || IEquals(mode, "0") || IEquals(mode, "off") || IEquals(mode, "none")) {
    g_runtime_config = cfg;
    return;
  }

  if (IEquals(mode, "checker") || IEquals(mode, "checkerboard")) {
    cfg.mode = InputZeroDitherMode::kCheckerboard;
  } else if (IEquals(mode, "random") || IEquals(mode, "rand")) {
    cfg.mode = InputZeroDitherMode::kRandom;
  } else {
    TVM_FFI_THROW(ValueError) << "Invalid input zero dither mode: " << mode;
  }

  cfg.enabled = true;
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS0", "GLOW_INPUT_ZERO_DITHER_EPS0", &cfg.eps_min);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS1", "GLOW_INPUT_ZERO_DITHER_EPS1", &cfg.eps_max);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS_MIN", "GLOW_INPUT_ZERO_DITHER_EPS_MIN",
                &cfg.eps_min);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_EPS_MAX", "GLOW_INPUT_ZERO_DITHER_EPS_MAX",
                &cfg.eps_max);
  ParseFloatEnv("TVM_INPUT_ZERO_DITHER_THRESH", "GLOW_INPUT_ZERO_DITHER_THRESH",
                &cfg.zero_threshold);
  cfg.has_seed = ParseUint32Env("TVM_INPUT_ZERO_DITHER_SEED", "GLOW_INPUT_ZERO_DITHER_SEED",
                                &cfg.seed);

  const char* layout =
      GetEnvWithFallback("TVM_INPUT_ZERO_DITHER_LAYOUT", "GLOW_INPUT_ZERO_DITHER_LAYOUT");
  if (layout != nullptr) {
    cfg.layout = ParseLayoutOrThrow(layout);
  }

  TVM_FFI_CHECK(cfg.zero_threshold >= 0.0f, ValueError)
      << "Input zero dither threshold must be >= 0";
  TVM_FFI_CHECK(cfg.eps_min >= 0.0f && cfg.eps_max >= 0.0f, ValueError)
      << "Input zero dither eps range must be >= 0";
  TVM_FFI_CHECK(cfg.eps_min <= cfg.eps_max, ValueError)
      << "Input zero dither eps_min must be <= eps_max";

  g_runtime_config = cfg;
}

inline const RuntimeConfig& Config() {
  std::call_once(g_runtime_config_init_flag, InitializeRuntimeConfig);
  return g_runtime_config;
}

inline bool IsFloat32Tensor(const DLTensor* tensor) {
  return tensor != nullptr && tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32 &&
         tensor->dtype.lanes == 1;
}

inline bool CanDitherTensor(const DLTensor* tensor) {
  return tensor != nullptr && IsFloat32Tensor(tensor) && tensor->device.device_type == kDLCPU &&
         tensor->ndim == 4 && ffi::IsContiguous(*tensor);
}

inline const float* TensorData(const DLTensor* tensor) {
  const char* base = static_cast<const char*>(tensor->data);
  return reinterpret_cast<const float*>(base + tensor->byte_offset);
}

inline float* TensorData(DLTensor* tensor) {
  char* base = static_cast<char*>(tensor->data);
  return reinterpret_cast<float*>(base + tensor->byte_offset);
}

uint32_t MixHash(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

uint32_t MakeHashSeed(const RuntimeConfig& cfg, size_t linear_index, size_t y, size_t x) {
  uint64_t index64 = static_cast<uint64_t>(linear_index);
  uint32_t seed = cfg.has_seed ? cfg.seed : 0x6d2b79f5U;
  seed ^= static_cast<uint32_t>(index64);
  seed ^= static_cast<uint32_t>(index64 >> 32) * 0x9e3779b9U;
  seed ^= static_cast<uint32_t>(y) * 0x85ebca6bU;
  seed ^= static_cast<uint32_t>(x) * 0xc2b2ae35U;
  return MixHash(seed);
}

float ComputeDelta(const RuntimeConfig& cfg, size_t linear_index, size_t y, size_t x) {
  if (cfg.mode == InputZeroDitherMode::kCheckerboard) {
    return ((y + x) & 1U) == 0 ? cfg.eps_min : cfg.eps_max;
  }
  uint32_t hash = MakeHashSeed(cfg, linear_index, y, x);
  float unit = static_cast<float>(hash >> 8) * (1.0f / 16777216.0f);
  return cfg.eps_min + (cfg.eps_max - cfg.eps_min) * unit;
}

float DitherValue(float value, size_t linear_index, size_t y, size_t x, const RuntimeConfig& cfg,
                  size_t* changed) {
  if (std::fabs(value) > cfg.zero_threshold) {
    return value;
  }
  if (changed != nullptr) {
    ++(*changed);
  }
  return value + ComputeDelta(cfg, linear_index, y, x);
}

void LogSummary(const RuntimeConfig& cfg, size_t changed, size_t total) {
  if (cfg.silent) {
    return;
  }
  std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10);
  if (cfg.mode == InputZeroDitherMode::kCheckerboard) {
    std::cerr << "input_zero_dither=checkerboard"
              << " layout=" << (cfg.layout == InputZeroDitherLayout::kNCHW ? "NCHW" : "NHWC")
              << " eps0=" << cfg.eps_min << " eps1=" << cfg.eps_max
              << " thresh=" << cfg.zero_threshold << " changed=" << changed << "/" << total;
  } else {
    std::cerr << "input_zero_dither=random"
              << " layout=" << (cfg.layout == InputZeroDitherLayout::kNCHW ? "NCHW" : "NHWC")
              << " eps_min=" << cfg.eps_min << " eps_max=" << cfg.eps_max
              << " thresh=" << cfg.zero_threshold << " changed=" << changed << "/" << total;
    if (cfg.has_seed) {
      std::cerr << " seed=" << cfg.seed;
    } else {
      std::cerr << " seed=default_hash";
    }
  }
  std::cerr << "\n";
}

void CopyWithDither(const float* src, float* dst, const DLTensor* tensor, const RuntimeConfig& cfg) {
  int64_t n = tensor->shape[0];
  int64_t c = cfg.layout == InputZeroDitherLayout::kNCHW ? tensor->shape[1] : tensor->shape[3];
  int64_t h = cfg.layout == InputZeroDitherLayout::kNCHW ? tensor->shape[2] : tensor->shape[1];
  int64_t w = cfg.layout == InputZeroDitherLayout::kNCHW ? tensor->shape[3] : tensor->shape[2];
  size_t changed = 0;

  if (cfg.layout == InputZeroDitherLayout::kNCHW) {
    for (int64_t ni = 0; ni < n; ++ni) {
      for (int64_t ci = 0; ci < c; ++ci) {
        for (int64_t yi = 0; yi < h; ++yi) {
          for (int64_t xi = 0; xi < w; ++xi) {
            size_t index = static_cast<size_t>(((ni * c + ci) * h + yi) * w + xi);
            dst[index] = DitherValue(src[index], index, static_cast<size_t>(yi),
                                     static_cast<size_t>(xi), cfg, &changed);
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
            dst[index] = DitherValue(src[index], index, static_cast<size_t>(yi),
                                     static_cast<size_t>(xi), cfg, &changed);
          }
        }
      }
    }
  }

  LogSummary(cfg, changed, static_cast<size_t>(n * c * h * w));
}

}  // namespace

bool IsEnabled() { return Config().enabled; }

bool ShouldApply(const DLTensor* tensor) { return IsEnabled() && CanDitherTensor(tensor); }

bool TryCopyFromBytes(const void* src_data, size_t nbytes, DLTensor* dst) {
  if (!ShouldApply(dst)) {
    return false;
  }
  TVM_FFI_ICHECK_EQ(GetDataSize(*dst), nbytes);
  CopyWithDither(static_cast<const float*>(src_data), TensorData(dst), dst, Config());
  return true;
}

bool TryCopyFromTensor(const DLTensor* src, DLTensor* dst) {
  if (!ShouldApply(src) || !CanDitherTensor(dst)) {
    return false;
  }
  TVM_FFI_ICHECK_EQ(src->ndim, dst->ndim);
  TVM_FFI_ICHECK_EQ(src->dtype.code, dst->dtype.code);
  TVM_FFI_ICHECK_EQ(src->dtype.bits, dst->dtype.bits);
  TVM_FFI_ICHECK_EQ(src->dtype.lanes, dst->dtype.lanes);
  for (int i = 0; i < src->ndim; ++i) {
    TVM_FFI_ICHECK_EQ(src->shape[i], dst->shape[i]);
  }
  CopyWithDither(TensorData(src), TensorData(dst), dst, Config());
  return true;
}

}  // namespace inputzerodither
}  // namespace runtime
}  // namespace tvm
