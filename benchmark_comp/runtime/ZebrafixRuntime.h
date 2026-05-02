#ifndef BENCHMARK_RUNTIME_ZEBRAFIX_RUNTIME_H
#define BENCHMARK_RUNTIME_ZEBRAFIX_RUNTIME_H

#include <cstddef>
#include <cstdint>

extern "C" {

uint64_t zebrafixExpandedSizeBytes(uint64_t logical_size);

void zebrafixClearRanges();
void zebrafixRegisterRange(void *logical_base, uint64_t logical_size);
void zebrafixUnregisterRange(void *logical_base);

void zebrafixLoadBytes(void *dst, const void *logical_addr, uint64_t size);
void zebrafixStoreBytes(void *logical_addr, const void *src, uint64_t size);
void *zebrafixLookupShadowBase(const void *logical_base);
uint64_t zebrafixLookupLogicalOffset(const void *logical_base);
void zebrafixLoad4Cached(void *dst, const void *shadow_base,
                         uint64_t logical_offset);
void zebrafixLoad8Cached(void *dst, const void *shadow_base,
                         uint64_t logical_offset);
void zebrafixLoad16Cached(void *dst, const void *shadow_base,
                          uint64_t logical_offset);
uint32_t zebrafixLoad32Maybe(const void *logical_addr);
uint64_t zebrafixLoad64Maybe(const void *logical_addr);
unsigned __int128 zebrafixLoad128Maybe(const void *logical_addr);
uint32_t zebrafixLoad32Value(const void *shadow_base, uint64_t logical_offset);
uint64_t zebrafixLoad64Value(const void *shadow_base, uint64_t logical_offset);
unsigned __int128 zebrafixLoad128Value(const void *shadow_base,
                                       uint64_t logical_offset);
void zebrafixLoadBytesCached(void *dst, const void *shadow_base,
                             uint64_t logical_offset, uint64_t size);
void zebrafixStore4Cached(void *shadow_base, uint64_t logical_offset,
                          const void *src);
void zebrafixStore8Cached(void *shadow_base, uint64_t logical_offset,
                          const void *src);
void zebrafixStore16Cached(void *shadow_base, uint64_t logical_offset,
                           const void *src);
void zebrafixStore32Maybe(void *logical_addr, uint32_t value);
void zebrafixStore64Maybe(void *logical_addr, uint64_t value);
void zebrafixStore128Maybe(void *logical_addr, unsigned __int128 value);
void zebrafixStore32Value(void *shadow_base, uint64_t logical_offset,
                          uint32_t value);
void zebrafixStore64Value(void *shadow_base, uint64_t logical_offset,
                          uint64_t value);
void zebrafixStore128Value(void *shadow_base, uint64_t logical_offset,
                           unsigned __int128 value);
void zebrafixStoreBytesCached(void *shadow_base, uint64_t logical_offset,
                              const void *src, uint64_t size);
void zebrafixMemcpy(void *dst, const void *src, uint64_t size);
void zebrafixMemmove(void *dst, const void *src, uint64_t size);
void zebrafixMemset(void *dst, uint8_t value, uint64_t size);

}

#endif
