/**
 * ReLU low-bit writeback patch helpers (experimental).
 *
 * This header provides a single, process-wide tag pointer plus a single
 * environment-variable gate for enabling/disabling the patch.
 *
 * The patch is considered enabled iff:
 *   GLOW_RELU_LOW12_PATCH=1
 *
 * Runtime patch parameters:
 *   GLOW_RELU_PATCH_BITS=1..31          (default: 12)
 *   GLOW_RELU_PATCH_FIXED_BITS=1..N     (default: 1)
 *   GLOW_RELU_PATCH_FIXED_VALUE=<uint32>
 *                                      (default: 1, interpreted as an
 *                                       M-bit value for bits [N-1, N-M])
 *   GLOW_RELU_PATCH_INC=<uint32>        (default: 3)
 *   GLOW_RELU_PATCH_POSITIVE=0|1        (default: 1)
 *
 * The tag pointer is used via Instruction::setUserData()/getUserData() to
 * precisely mark which IR instructions should take the patched execution path.
 */
#ifndef GLOW_SUPPORT_RELULOW12PATCH_H
#define GLOW_SUPPORT_RELULOW12PATCH_H

#include <cstdint>
#include <cstring>

namespace glow {
namespace relulow12 {

/// Returns true if the patch is enabled for this process.
bool isEnabled();

/// Returns a unique, stable tag pointer to be stored in Instruction userData.
void *tagPtr();
/// Returns a unique, stable tag pointer for lowered ReLU6 ElementMax IR.
void *relu6TagPtr();
/// Reseed the per-process patch counter for a new inference.
void resetInferencePatchSeed();
/// Returns the next per-address patched seed bits for this inference.
uint32_t nextPatchedSeedBits();

/// Default number of patch-window bits replaced in the float32 payload.
/// The window covers bits [0, N-1] and therefore can span the full mantissa
/// plus exponent field, while leaving the sign bit untouched.
constexpr uint32_t kDefaultBits = 12;
constexpr uint32_t kMinBits = 1;
constexpr uint32_t kMaxBits = 31;
constexpr uint32_t kNegativeZeroBits = 0x80000000u;
constexpr uint32_t kExpMask = 0x7f800000u;
constexpr uint32_t kMantissaMask = 0x007fffffu;
constexpr uint32_t kDefaultFixedBits = 1u;
constexpr uint32_t kDefaultFixedValue = 1u;
constexpr uint32_t kDefaultIncrement = 3u;
constexpr uint32_t kSeedDerivationStep = 3u;
constexpr uint32_t kSixBits = 0x40c00000u;

/// Backend-private fused-activation code for the patched ReLU semantics.
constexpr int32_t kLibjitReluLow12ActType = 101;
/// Backend-private fused-activation code for the patched ReLU6 semantics.
constexpr int32_t kLibjitRelu6Low12ActType = 102;

struct RuntimeConfig {
  uint32_t bitCount;
  uint32_t fixedBitCount;
  uint32_t patchPayloadBitCount;
  uint32_t lowMask;
  uint32_t highMask;
  uint32_t patchPayloadMask;
  uint32_t fixedWindowBits;
  uint32_t increment;
  uint32_t patchPositiveOutputs;
};

/// Initialized when the patch enable gate is evaluated. Hot paths only read it.
extern RuntimeConfig gRuntimeConfig;

inline uint32_t floatToBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float bitsToFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline bool isNaNBits(uint32_t bits) {
  return (bits & kExpMask) == kExpMask && (bits & kMantissaMask) != 0;
}

inline const RuntimeConfig &runtimeConfig() { return gRuntimeConfig; }

inline uint32_t bitCount() { return runtimeConfig().bitCount; }

inline uint32_t lowBitMask(uint32_t numBits) {
  return numBits == 0u
             ? 0u
             : static_cast<uint32_t>((uint64_t{1} << numBits) - 1u);
}

inline uint32_t mask() { return runtimeConfig().lowMask; }

inline uint32_t highMask() { return runtimeConfig().highMask; }

inline uint32_t fixedBitCount() { return runtimeConfig().fixedBitCount; }

inline uint32_t patchPayloadBitCount() {
  return runtimeConfig().patchPayloadBitCount;
}

inline uint32_t patchPayloadMask() { return runtimeConfig().patchPayloadMask; }

inline uint32_t fixedValueMask() { return lowBitMask(fixedBitCount()); }

inline uint32_t fixedValue() {
  return runtimeConfig().fixedWindowBits >> patchPayloadBitCount();
}

inline uint32_t increment() { return runtimeConfig().increment; }

inline bool patchPositiveOutputs() {
  return runtimeConfig().patchPositiveOutputs != 0u;
}

inline uint32_t normalizeNegativeZeroBits(uint32_t bits) {
  return bits == kNegativeZeroBits ? 0u : bits;
}

inline uint32_t reluResultBitsFromInputBits(uint32_t inputBits) {
  uint32_t reluBits = normalizeNegativeZeroBits(inputBits);
  if (isNaNBits(reluBits)) {
    return reluBits;
  }
  if (reluBits & kNegativeZeroBits) {
    return 0u;
  }
  return reluBits;
}

inline uint32_t patchedWindowBitsFromOldBitsFallback(uint32_t oldBits) {
  const auto &cfg = runtimeConfig();
  return cfg.fixedWindowBits +
         (((oldBits & cfg.patchPayloadMask) + cfg.increment) &
          cfg.patchPayloadMask);
}

inline uint32_t applyPatchedWindowToBitsFallback(uint32_t baseBits,
                                                 uint32_t oldBits) {
  const auto &cfg = runtimeConfig();
  return (baseBits & cfg.highMask) | patchedWindowBitsFromOldBitsFallback(oldBits);
}

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
inline uint32_t applyPatchedWindowToBitsAsm(uint32_t baseBits,
                                            uint32_t oldBits) {
  const auto &cfg = runtimeConfig();
  uint32_t patchedBits;
  uint32_t baseHighBits;
  asm volatile(
      "movl %[old_bits], %[patched_bits]\n\t"
      "andl %[payload_mask], %[patched_bits]\n\t"
      "addl %[increment], %[patched_bits]\n\t"
      "andl %[payload_mask], %[patched_bits]\n\t"
      "orl %[fixed_window_bits], %[patched_bits]\n\t"
      "movl %[base_bits], %[base_high_bits]\n\t"
      "andl %[high_mask], %[base_high_bits]\n\t"
      "orl %[base_high_bits], %[patched_bits]\n\t"
      : [patched_bits] "=&r"(patchedBits),
        [base_high_bits] "=&r"(baseHighBits)
      : [old_bits] "r"(oldBits),
        [payload_mask] "r"(cfg.patchPayloadMask),
        [increment] "r"(cfg.increment),
        [fixed_window_bits] "r"(cfg.fixedWindowBits),
        [base_bits] "r"(baseBits),
        [high_mask] "r"(cfg.highMask)
      : "cc");
  return patchedBits;
}
#endif

inline uint32_t applyPatchedWindowToBits(uint32_t baseBits, uint32_t oldBits) {
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  return applyPatchedWindowToBitsAsm(baseBits, oldBits);
#else
  return applyPatchedWindowToBitsFallback(baseBits, oldBits);
#endif
}

inline bool isPatchedZeroWindowBits(uint32_t bits) {
  const auto &cfg = runtimeConfig();
  if ((bits & cfg.highMask) != 0u) {
    return false;
  }
  return (bits & ~cfg.patchPayloadMask) == cfg.fixedWindowBits;
}

inline bool hasPatchedPayloadSignature(uint32_t bits) {
  const auto &cfg = runtimeConfig();
  return (bits & (cfg.lowMask & ~cfg.patchPayloadMask)) == cfg.fixedWindowBits;
}

inline uint32_t selectPatchedSeedBits(uint32_t baseBits, uint32_t oldBits) {
  if (baseBits == 0u) {
    return isPatchedZeroWindowBits(oldBits) ? oldBits : nextPatchedSeedBits();
  }
  return hasPatchedPayloadSignature(oldBits) ? oldBits : nextPatchedSeedBits();
}

inline uint32_t patchedReluBitsFromInputBitsAndOldBits(uint32_t inputBits,
                                                       uint32_t oldBits) {
  const uint32_t reluBits = reluResultBitsFromInputBits(inputBits);
  if (isNaNBits(reluBits)) {
    return reluBits;
  }

  if (!patchPositiveOutputs() && reluBits != 0u) {
    return reluBits;
  }

  return applyPatchedWindowToBits(reluBits,
                                  selectPatchedSeedBits(reluBits, oldBits));
}

inline float patchedReluFloat(float input, float oldOutput) {
  return bitsToFloat(
      patchedReluBitsFromInputBitsAndOldBits(floatToBits(input),
                                             floatToBits(oldOutput)));
}

inline float relu6Float(float input) {
  if (!(input > 0.0f)) {
    return 0.0f;
  }
  return input > 6.0f ? 6.0f : input;
}

inline uint32_t relu6ResultBitsFromInputBits(uint32_t inputBits) {
  if (isNaNBits(inputBits)) {
    return 0u;
  }

  const uint32_t reluBits = normalizeNegativeZeroBits(inputBits);
  if (reluBits == 0u) {
    return 0u;
  }

  if (reluBits & kNegativeZeroBits) {
    return 0u;
  }

  return reluBits > kSixBits ? kSixBits : reluBits;
}

inline uint32_t patchedRelu6BitsFromInputBitsAndOldBits(uint32_t inputBits,
                                                        uint32_t oldBits) {
  const uint32_t relu6Bits = relu6ResultBitsFromInputBits(inputBits);
  if (relu6Bits != 0u) {
    return relu6Bits;
  }

  return applyPatchedWindowToBits(0u, selectPatchedSeedBits(0u, oldBits));
}

inline float patchedRelu6FloatZeroOnly(float input, float oldOutput) {
  // Keep the zero-clamp branch in bit space so helper-local float 0.0f does
  // not need to materialize on the stack before patching.
  return bitsToFloat(
      patchedRelu6BitsFromInputBitsAndOldBits(floatToBits(input),
                                              floatToBits(oldOutput)));
}

} // namespace relulow12
} // namespace glow

#endif // GLOW_SUPPORT_RELULOW12PATCH_H
