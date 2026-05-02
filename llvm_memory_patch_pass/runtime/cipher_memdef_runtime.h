#ifndef CIPHER_MEMDEF_RUNTIME_H_
#define CIPHER_MEMDEF_RUNTIME_H_

#include <stdint.h>

#if defined(_WIN32)
#define CIPHER_MEMDEF_EXPORT __declspec(dllexport)
#else
#define CIPHER_MEMDEF_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct CipherMemdefConfig {
  uint32_t bit_count;
  uint32_t fixed_bit_count;
  uint32_t patch_payload_bit_count;
  uint32_t low_mask;
  uint32_t high_mask;
  uint32_t patch_payload_mask;
  uint32_t fixed_window_bits;
  uint32_t increment;
  uint32_t patch_positive_outputs;
  uint32_t signature_mask;
  uint32_t global_seed;
};

CIPHER_MEMDEF_EXPORT extern struct CipherMemdefConfig __cipher_memdef_config;
CIPHER_MEMDEF_EXPORT void cipher_memdef_reset_seed(uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif
