#include "cipher_memdef_runtime.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

CIPHER_MEMDEF_EXPORT struct CipherMemdefConfig __cipher_memdef_config = {
    0u, 0u, 0u, 0u, UINT32_MAX, 0u, 0u, 0u, 0u, 0u, 0u};

_Static_assert(offsetof(struct CipherMemdefConfig, low_mask) == 12,
               "MIR pass offset mismatch: low_mask");
_Static_assert(offsetof(struct CipherMemdefConfig, high_mask) == 16,
               "MIR pass offset mismatch: high_mask");
_Static_assert(offsetof(struct CipherMemdefConfig, patch_payload_mask) == 20,
               "MIR pass offset mismatch: patch_payload_mask");
_Static_assert(offsetof(struct CipherMemdefConfig, fixed_window_bits) == 24,
               "MIR pass offset mismatch: fixed_window_bits");
_Static_assert(offsetof(struct CipherMemdefConfig, increment) == 28,
               "MIR pass offset mismatch: increment");
_Static_assert(offsetof(struct CipherMemdefConfig, signature_mask) == 36,
               "MIR pass offset mismatch: signature_mask");
_Static_assert(offsetof(struct CipherMemdefConfig, global_seed) == 40,
               "MIR pass offset mismatch: global_seed");

static uint32_t low_mask(uint32_t bits) {
  if (bits == 0u) {
    return 0u;
  }
  if (bits >= 32u) {
    return UINT32_MAX;
  }
  return (uint32_t)((UINT64_C(1) << bits) - 1u);
}

static uint32_t parse_u32(const char *name, uint32_t fallback,
                          uint32_t min_value, uint32_t max_value) {
  const char *raw = getenv(name);
  if (raw == 0 || raw[0] == '\0') {
    return fallback;
  }

  char *end = 0;
  unsigned long parsed = strtoul(raw, &end, 0);
  if (end == raw || *end != '\0' || parsed > (unsigned long)UINT32_MAX) {
    return fallback;
  }

  uint32_t value = (uint32_t)parsed;
  if (value < min_value || value > max_value) {
    return fallback;
  }
  return value;
}

static uint32_t mix_u32(uint32_t value) {
  value ^= value >> 16;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15;
  value *= UINT32_C(0x846ca68b);
  value ^= value >> 16;
  return value;
}

static uint32_t default_seed(void) {
  uintptr_t address = (uintptr_t)&__cipher_memdef_config;
  uint32_t value = (uint32_t)time(NULL);
  value ^= (uint32_t)clock() * UINT32_C(0x9e3779b9);
  value ^= (uint32_t)address;
  value ^= (uint32_t)(address >> 32);
  return mix_u32(value);
}

static void cipher_memdef_init(void) {
  struct CipherMemdefConfig cfg;

  cfg.bit_count = parse_u32("CIPHER_MEMDEF_BITS", 0u, 0u, 31u);
  cfg.fixed_bit_count =
      parse_u32("CIPHER_MEMDEF_FIXED_BITS", 0u, 0u, cfg.bit_count);
  cfg.patch_payload_bit_count = cfg.bit_count - cfg.fixed_bit_count;

  cfg.low_mask = low_mask(cfg.bit_count);
  cfg.high_mask = ~cfg.low_mask;
  cfg.patch_payload_mask = low_mask(cfg.patch_payload_bit_count);
  cfg.signature_mask = cfg.low_mask & ~cfg.patch_payload_mask;

  uint32_t fixed_mask = low_mask(cfg.fixed_bit_count);
  uint32_t fixed_value =
      parse_u32("CIPHER_MEMDEF_FIXED_VALUE", 0u, 0u, fixed_mask);
  cfg.fixed_window_bits = fixed_value << cfg.patch_payload_bit_count;

  cfg.increment =
      parse_u32("CIPHER_MEMDEF_INCREMENT", 0u, 0u, UINT32_MAX);
  cfg.patch_positive_outputs =
      parse_u32("CIPHER_MEMDEF_PATCH_POSITIVE", 0u, 0u, 1u);
  cfg.global_seed =
      parse_u32("CIPHER_MEMDEF_SEED", default_seed(), 0u, UINT32_MAX);

  __cipher_memdef_config = cfg;
}

CIPHER_MEMDEF_EXPORT void cipher_memdef_reset_seed(uint32_t seed) {
  __cipher_memdef_config.global_seed = seed;
}

#if defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) static void (*cipher_memdef_init_ref)(void) =
    cipher_memdef_init;
#else
__attribute__((constructor)) static void cipher_memdef_ctor(void) {
  cipher_memdef_init();
}
#endif
