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
 * \file relu_low12_patch.cc
 * \brief Low-bit ReLU/ReLU6 writeback patch helpers.
 */

#include <tvm/runtime/relu_low12_patch.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>

namespace tvm {
namespace runtime {
namespace relulow12 {
namespace {

constexpr uint32_t kDefaultBits = 12u;
constexpr uint32_t kMinBits = 1u;
constexpr uint32_t kMaxBits = 31u;
constexpr uint32_t kNegativeZeroBits = 0x80000000u;
constexpr uint32_t kExpMask = 0x7f800000u;
constexpr uint32_t kMantissaMask = 0x007fffffu;
constexpr uint32_t kDefaultFixedBits = 1u;
constexpr uint32_t kDefaultFixedValue = 1u;
constexpr uint32_t kDefaultIncrement = 3u;
constexpr uint32_t kSeedDerivationStep = 3u;
constexpr uint32_t kSixBits = 0x40c00000u;

struct RuntimeConfig {
  uint32_t bit_count;
  uint32_t fixed_bit_count;
  uint32_t patch_payload_bit_count;
  uint32_t low_mask;
  uint32_t high_mask;
  uint32_t patch_payload_mask;
  uint32_t fixed_window_bits;
  uint32_t increment;
  uint32_t patch_positive_outputs;
};

inline uint32_t LowBitMask(uint32_t num_bits) {
  return num_bits == 0u ? 0u : static_cast<uint32_t>((uint64_t{1} << num_bits) - 1u);
}

RuntimeConfig g_runtime_config{
    kDefaultBits,
    kDefaultFixedBits,
    kDefaultBits - kDefaultFixedBits,
    LowBitMask(kDefaultBits),
    ~LowBitMask(kDefaultBits),
    LowBitMask(kDefaultBits - kDefaultFixedBits),
    kDefaultFixedValue << (kDefaultBits - kDefaultFixedBits),
    kDefaultIncrement,
    1u,
};
std::once_flag g_runtime_config_init_flag;
std::atomic<uint32_t> g_patch_seed_counter{0u};

inline uint32_t FloatToBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float BitsToFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline bool IsNaNBits(uint32_t bits) {
  return (bits & kExpMask) == kExpMask && (bits & kMantissaMask) != 0u;
}

uint32_t ParseEnvUint32(const char* name, uint32_t default_value, uint32_t min_value,
                       uint32_t max_value) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  }

  char* end = nullptr;
  unsigned long parsed = std::strtoul(env, &end, 0);
  if (end == nullptr || end[0] != '\0') {
    return default_value;
  }
  if (parsed < min_value || parsed > max_value) {
    return default_value;
  }
  return static_cast<uint32_t>(parsed);
}

void InitializeRuntimeConfig() {
  RuntimeConfig cfg = g_runtime_config;
  cfg.bit_count = ParseEnvUint32("TVM_RELU_PATCH_BITS", kDefaultBits, kMinBits, kMaxBits);
  cfg.fixed_bit_count =
      ParseEnvUint32("TVM_RELU_PATCH_FIXED_BITS", kDefaultFixedBits, 1u, cfg.bit_count);
  cfg.patch_payload_bit_count = cfg.bit_count - cfg.fixed_bit_count;
  cfg.low_mask = LowBitMask(cfg.bit_count);
  cfg.high_mask = ~cfg.low_mask;
  cfg.patch_payload_mask = LowBitMask(cfg.patch_payload_bit_count);
  uint32_t fixed_value_mask = LowBitMask(cfg.fixed_bit_count);
  uint32_t fixed_value = ParseEnvUint32("TVM_RELU_PATCH_FIXED_VALUE",
                                        kDefaultFixedValue & fixed_value_mask, 0u,
                                        fixed_value_mask);
  cfg.fixed_window_bits = fixed_value << cfg.patch_payload_bit_count;
  cfg.increment = ParseEnvUint32("TVM_RELU_PATCH_INC", kDefaultIncrement, 0u,
                                 std::numeric_limits<uint32_t>::max());
  cfg.patch_positive_outputs =
      ParseEnvUint32("TVM_RELU_PATCH_POSITIVE", 1u, 0u, 1u);
  g_runtime_config = cfg;

  std::random_device rd;
  uint32_t random_seed = static_cast<uint32_t>(rd());
  random_seed ^= static_cast<uint32_t>(rd()) * 0x9e3779b9u;
  g_patch_seed_counter.store(random_seed & cfg.patch_payload_mask, std::memory_order_release);
}

inline const RuntimeConfig& Config() { return g_runtime_config; }

inline uint32_t NormalizeNegativeZeroBits(uint32_t bits) {
  return bits == kNegativeZeroBits ? 0u : bits;
}

inline uint32_t ReluResultBitsFromInputBits(uint32_t input_bits) {
  uint32_t relu_bits = NormalizeNegativeZeroBits(input_bits);
  if (IsNaNBits(relu_bits)) {
    return relu_bits;
  }
  if (relu_bits & kNegativeZeroBits) {
    return 0u;
  }
  return relu_bits;
}

inline uint32_t PatchedWindowBitsFromOldBits(uint32_t old_bits) {
  const RuntimeConfig& cfg = Config();
  return cfg.fixed_window_bits +
         (((old_bits & cfg.patch_payload_mask) + cfg.increment) & cfg.patch_payload_mask);
}

inline bool HasPatchedPayloadSignature(uint32_t bits) {
  const RuntimeConfig& cfg = Config();
  return (bits & (cfg.low_mask & ~cfg.patch_payload_mask)) == cfg.fixed_window_bits;
}

inline uint32_t ApplyPatchedWindowToBitsFallback(uint32_t base_bits, uint32_t old_bits) {
  const RuntimeConfig& cfg = Config();
  return (base_bits & cfg.high_mask) | PatchedWindowBitsFromOldBits(old_bits);
}

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
inline uint32_t ApplyPatchedWindowToBitsAsm(uint32_t base_bits, uint32_t old_bits) {
  const RuntimeConfig& cfg = Config();
  uint32_t patched_bits;
  uint32_t base_high_bits;
  asm volatile(
      "movl %[old_bits], %[patched_bits]\n\t"
      "andl %[payload_mask], %[patched_bits]\n\t"
      "addl %[increment], %[patched_bits]\n\t"
      "andl %[payload_mask], %[patched_bits]\n\t"
      "orl %[fixed_window_bits], %[patched_bits]\n\t"
      "movl %[base_bits], %[base_high_bits]\n\t"
      "andl %[high_mask], %[base_high_bits]\n\t"
      "orl %[base_high_bits], %[patched_bits]\n\t"
      : [patched_bits] "=&r"(patched_bits), [base_high_bits] "=&r"(base_high_bits)
      : [old_bits] "r"(old_bits), [payload_mask] "r"(cfg.patch_payload_mask),
        [increment] "r"(cfg.increment), [fixed_window_bits] "r"(cfg.fixed_window_bits),
        [base_bits] "r"(base_bits), [high_mask] "r"(cfg.high_mask)
      : "cc");
  return patched_bits;
}
#endif

inline uint32_t ApplyPatchedWindowToBits(uint32_t base_bits, uint32_t old_bits) {
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  return ApplyPatchedWindowToBitsAsm(base_bits, old_bits);
#else
  return ApplyPatchedWindowToBitsFallback(base_bits, old_bits);
#endif
}

inline bool IsPatchedZeroWindowBits(uint32_t bits) {
  const RuntimeConfig& cfg = Config();
  if ((bits & cfg.high_mask) != 0u) {
    return false;
  }
  return (bits & ~cfg.patch_payload_mask) == cfg.fixed_window_bits;
}

inline uint32_t NextPatchedSeedBits() {
  const RuntimeConfig& cfg = Config();
  uint32_t payload =
      g_patch_seed_counter.fetch_add(kSeedDerivationStep, std::memory_order_relaxed);
  return cfg.fixed_window_bits | (payload & cfg.patch_payload_mask);
}

inline uint32_t SelectPatchedSeedBits(uint32_t base_bits, uint32_t old_bits) {
  if (base_bits == 0u) {
    return IsPatchedZeroWindowBits(old_bits) ? old_bits : NextPatchedSeedBits();
  }
  return HasPatchedPayloadSignature(old_bits) ? old_bits : NextPatchedSeedBits();
}

inline uint32_t PatchedReluBitsFromInputBitsAndOldBits(uint32_t input_bits, uint32_t old_bits) {
  uint32_t relu_bits = ReluResultBitsFromInputBits(input_bits);
  if (IsNaNBits(relu_bits)) {
    return relu_bits;
  }

  if (Config().patch_positive_outputs == 0u && relu_bits != 0u) {
    return relu_bits;
  }

  return ApplyPatchedWindowToBits(relu_bits, SelectPatchedSeedBits(relu_bits, old_bits));
}

inline uint32_t Relu6ResultBitsFromInputBits(uint32_t input_bits) {
  if (IsNaNBits(input_bits)) {
    return 0u;
  }

  uint32_t relu_bits = NormalizeNegativeZeroBits(input_bits);
  if (relu_bits == 0u) {
    return 0u;
  }

  if (relu_bits & kNegativeZeroBits) {
    return 0u;
  }

  return relu_bits > kSixBits ? kSixBits : relu_bits;
}

inline uint32_t PatchedRelu6BitsFromInputBitsAndOldBits(uint32_t input_bits, uint32_t old_bits) {
  uint32_t relu6_bits = Relu6ResultBitsFromInputBits(input_bits);
  if (relu6_bits != 0u) {
    return relu6_bits;
  }

  return ApplyPatchedWindowToBits(0u, SelectPatchedSeedBits(0u, old_bits));
}

}  // namespace

bool IsEnabled() {
  static std::atomic<int> cached{-1};
  int value = cached.load(std::memory_order_acquire);
  if (value != -1) {
    return value == 1;
  }

  const char* env = std::getenv("TVM_RELU_LOW12_PATCH");
  bool enabled = env != nullptr && env[0] == '1' && env[1] == '\0';
  if (enabled) {
    std::call_once(g_runtime_config_init_flag, InitializeRuntimeConfig);
  }
  cached.store(enabled ? 1 : 0, std::memory_order_release);
  return enabled;
}

namespace {

struct PatchConfigInitializer {
  PatchConfigInitializer() { (void)IsEnabled(); }
};

PatchConfigInitializer g_patch_config_initializer;

}  // namespace

void ResetInferencePatchSeed() {
  if (!IsEnabled()) {
    return;
  }
  const RuntimeConfig& cfg = Config();
  std::random_device rd;
  uint32_t random_seed = static_cast<uint32_t>(rd());
  random_seed ^= static_cast<uint32_t>(rd()) * 0x9e3779b9u;
  g_patch_seed_counter.store(random_seed & cfg.patch_payload_mask, std::memory_order_release);
}

float PatchedReluFloat(float input, float old_output) {
  return BitsToFloat(
      PatchedReluBitsFromInputBitsAndOldBits(FloatToBits(input), FloatToBits(old_output)));
}

float PatchedRelu6FloatZeroOnly(float input, float old_output) {
  return BitsToFloat(
      PatchedRelu6BitsFromInputBitsAndOldBits(FloatToBits(input), FloatToBits(old_output)));
}

}  // namespace relulow12
}  // namespace runtime
}  // namespace tvm

extern "C" TVM_DLL float tvm_relu_low12_f32_scalar(float input, float old_output) {
  return tvm::runtime::relulow12::PatchedReluFloat(input, old_output);
}

extern "C" TVM_DLL float tvm_relu6_low12_f32_scalar(float input, float old_output) {
  return tvm::runtime::relulow12::PatchedRelu6FloatZeroOnly(input, old_output);
}

extern "C" TVM_DLL void tvm_relu_low12_reset_inference_seed(void) {
  tvm::runtime::relulow12::ResetInferencePatchSeed();
}
