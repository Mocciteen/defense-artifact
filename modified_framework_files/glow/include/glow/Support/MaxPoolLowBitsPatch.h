/**
 * MaxPool local-max same-writeback low-bit patch helpers (experimental).
 *
 * The patch is considered enabled iff:
 *   GLOW_MAXPOOL_LOWBIT_PATCH=1
 *
 * Runtime patch parameters:
 *   GLOW_MAXPOOL_PATCH_BITS=1..31   (default: 6)
 *   GLOW_MAXPOOL_PATCH_INC=<uint32> (default: 3)
 *
 * Semantics:
 *   when a MaxPool local-max update would rewrite the same float32 bit-pattern
 *   back to the local spill slot, replace baseBits[0, N-1] with
 *   ((oldBits[0, N-1] + inc) mod 2^N)
 */
#ifndef GLOW_SUPPORT_MAXPOOLLOWBITSPATCH_H
#define GLOW_SUPPORT_MAXPOOLLOWBITSPATCH_H

#include <cstdint>
#include <cstring>

namespace glow {
namespace maxpoollowbits {

bool isEnabled();

constexpr uint32_t kDefaultBits = 6u;
constexpr uint32_t kMinBits = 1u;
constexpr uint32_t kMaxBits = 31u;
constexpr uint32_t kDefaultIncrement = 3u;

struct RuntimeConfig {
  uint32_t bitCount;
  uint32_t lowMask;
  uint32_t highMask;
  uint32_t increment;
};

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

inline uint32_t lowBitMask(uint32_t numBits) {
  return numBits == 0u
             ? 0u
             : static_cast<uint32_t>((uint64_t{1} << numBits) - 1u);
}

inline const RuntimeConfig &runtimeConfig() { return gRuntimeConfig; }

inline bool sameWritebackBits(uint32_t nextBits, uint32_t oldBits) {
  return nextBits == oldBits;
}

inline bool sameWritebackFloat(float nextValue, float oldValue) {
  return sameWritebackBits(floatToBits(nextValue), floatToBits(oldValue));
}

inline uint32_t patchedLowBitsFromOldBitsFallback(uint32_t oldBits) {
  const auto &cfg = runtimeConfig();
  return ((oldBits & cfg.lowMask) + cfg.increment) & cfg.lowMask;
}

inline uint32_t applyPatchedLowBitsToBaseBitsFallback(uint32_t baseBits,
                                                      uint32_t oldBits) {
  const auto &cfg = runtimeConfig();
  return (baseBits & cfg.highMask) | patchedLowBitsFromOldBitsFallback(oldBits);
}

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
inline uint32_t applyPatchedLowBitsToBaseBitsAsm(uint32_t baseBits,
                                                 uint32_t oldBits) {
  const auto &cfg = runtimeConfig();
  uint32_t patchedBits;
  asm volatile(
      "movl %[old_bits], %[patched_bits]\n\t"
      "andl %[low_mask], %[patched_bits]\n\t"
      "addl %[increment], %[patched_bits]\n\t"
      "andl %[low_mask], %[patched_bits]\n\t"
      "andl %[high_mask], %[base_bits]\n\t"
      "orl %[patched_bits], %[base_bits]\n\t"
      : [base_bits] "+&r"(baseBits), [patched_bits] "=&r"(patchedBits)
      : [old_bits] "r"(oldBits), [low_mask] "r"(cfg.lowMask),
        [high_mask] "r"(cfg.highMask), [increment] "r"(cfg.increment)
      : "cc");
  return baseBits;
}
#endif

inline uint32_t applyPatchedLowBitsToBaseBits(uint32_t baseBits,
                                              uint32_t oldBits) {
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  return applyPatchedLowBitsToBaseBitsAsm(baseBits, oldBits);
#else
  return applyPatchedLowBitsToBaseBitsFallback(baseBits, oldBits);
#endif
}

inline float patchedFloatFromBaseAndOldBits(float baseValue, uint32_t oldBits) {
  return bitsToFloat(
      applyPatchedLowBitsToBaseBits(floatToBits(baseValue), oldBits));
}

} // namespace maxpoollowbits
} // namespace glow

#endif // GLOW_SUPPORT_MAXPOOLLOWBITSPATCH_H
