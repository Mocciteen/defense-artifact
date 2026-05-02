#include "ZebrafixRuntime.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t kLogicalGranuleBytes = 8;
constexpr uint64_t kShadowSlotBytes = 16;
constexpr uint64_t kCounterOffset = 8;

struct Range {
  uint8_t *base;
  uint8_t *end;
  uint64_t logicalSize;
  std::vector<uint8_t> shadow;
};

std::vector<Range> &ranges() {
  static std::vector<Range> instance;
  return instance;
}

std::atomic<uint64_t> &globalCounter() {
  static std::atomic<uint64_t> counter{1};
  return counter;
}

std::atomic<uint64_t> &rangesEpoch() {
  static std::atomic<uint64_t> epoch{1};
  return epoch;
}

bool mirrorRawStoreEnabled() {
  const char *value = std::getenv("ZEBRAFIX_DEBUG_MIRROR_RAW_STORE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

uint64_t shadowSlotCount(uint64_t logicalSize) {
  return (logicalSize + kLogicalGranuleBytes - 1) / kLogicalGranuleBytes;
}

uint64_t shadowByteOffset(uint64_t logicalOffset) {
  return (logicalOffset / kLogicalGranuleBytes) * kShadowSlotBytes +
         (logicalOffset % kLogicalGranuleBytes);
}

void sortRangesByBase() {
  auto &allRanges = ranges();
  std::sort(allRanges.begin(), allRanges.end(),
            [](const Range &lhs, const Range &rhs) { return lhs.base < rhs.base; });
}

bool rangesOverlap(const Range &range, const uint8_t *base, const uint8_t *end) {
  return base < range.end && range.base < end;
}

void bumpRangesEpoch() {
  rangesEpoch().fetch_add(1, std::memory_order_relaxed);
}

[[noreturn]] void abortOutOfBounds(const char *op, const Range &range,
                                   const void *addr, uint64_t size,
                                   uint64_t logicalOffset) {
  std::cerr << "zebrafix " << op << " out-of-bounds logical access base="
            << static_cast<const void *>(range.base) << " logical_size="
            << range.logicalSize << " addr=" << addr << " size=" << size
            << " logical_offset=" << logicalOffset << "\n";
  std::abort();
}

uint64_t checkedLogicalOffset(const Range &range, const void *addr, uint64_t size,
                              const char *op) {
  const auto logicalOffset =
      static_cast<uint64_t>(static_cast<const uint8_t *>(addr) - range.base);
  if (logicalOffset > range.logicalSize || size > range.logicalSize - logicalOffset) {
    abortOutOfBounds(op, range, addr, size, logicalOffset);
  }
  return logicalOffset;
}

Range *findRangeForAddress(const void *addr) {
  auto *byteAddr = static_cast<const uint8_t *>(addr);
  thread_local uint64_t cachedEpoch = 0;
  thread_local Range *cachedRange = nullptr;

  const uint64_t currentEpoch = rangesEpoch().load(std::memory_order_relaxed);
  if (cachedEpoch != currentEpoch) {
    cachedEpoch = currentEpoch;
    cachedRange = nullptr;
  }

  if (cachedRange != nullptr && byteAddr >= cachedRange->base &&
      byteAddr < cachedRange->end) {
    return cachedRange;
  }

  auto &allRanges = ranges();
  auto it = std::upper_bound(
      allRanges.begin(), allRanges.end(), byteAddr,
      [](const uint8_t *needle, const Range &range) { return needle < range.base; });
  if (it == allRanges.begin()) {
    return nullptr;
  }

  --it;
  if (byteAddr >= it->base && byteAddr < it->end) {
    cachedRange = &*it;
    return cachedRange;
  }
  return nullptr;
}

bool rangeContains(const Range &range, const void *addr, uint64_t size) {
  const auto *byteAddr = static_cast<const uint8_t *>(addr);
  return byteAddr >= range.base && byteAddr <= range.end &&
         size <= static_cast<uint64_t>(range.end - byteAddr);
}

const Range *findRangeForSpan(const void *addr, uint64_t size) {
  Range *range = findRangeForAddress(addr);
  if (range != nullptr && rangeContains(*range, addr, size)) {
    return range;
  }
  return nullptr;
}

Range *findMutableRangeForSpan(void *addr, uint64_t size) {
  Range *range = findRangeForAddress(addr);
  if (range != nullptr && rangeContains(*range, addr, size)) {
    return range;
  }
  return nullptr;
}

void loadProtectedBytes(void *dst, const Range &range, const void *src,
                        uint64_t size) {
  auto *out = static_cast<uint8_t *>(dst);
  uint64_t logicalOffset = checkedLogicalOffset(range, src, size, "load");
  uint64_t remaining = size;
  while (remaining > 0) {
    const uint64_t chunkIndex = logicalOffset / kLogicalGranuleBytes;
    const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
    const uint64_t chunkBytes =
        std::min<uint64_t>(remaining, kLogicalGranuleBytes - chunkOffset);
    const uint8_t *slotBase =
        range.shadow.data() +
        static_cast<std::ptrdiff_t>(chunkIndex * kShadowSlotBytes);
    std::memcpy(out, slotBase + chunkOffset, static_cast<std::size_t>(chunkBytes));
    out += chunkBytes;
    logicalOffset += chunkBytes;
    remaining -= chunkBytes;
  }
}

void loadProtectedBytesCached(void *dst, const uint8_t *shadowBase,
                              uint64_t logicalOffset, uint64_t size) {
  auto *out = static_cast<uint8_t *>(dst);
  uint64_t remaining = size;
  while (remaining > 0) {
    const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
    const uint64_t chunkBytes =
        std::min<uint64_t>(remaining, kLogicalGranuleBytes - chunkOffset);
    const uint8_t *slotBase =
        shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
    std::memcpy(out, slotBase + chunkOffset, static_cast<std::size_t>(chunkBytes));
    out += chunkBytes;
    logicalOffset += chunkBytes;
    remaining -= chunkBytes;
  }
}

inline void load4CachedFast(void *dst, const uint8_t *shadowBase,
                            uint64_t logicalOffset) {
  const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
  const uint8_t *slotBase =
      shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
  if (chunkOffset <= 4) {
    std::memcpy(dst, slotBase + chunkOffset, 4);
    return;
  }
  uint8_t *out = static_cast<uint8_t *>(dst);
  const uint64_t first = kLogicalGranuleBytes - chunkOffset;
  std::memcpy(out, slotBase + chunkOffset, static_cast<std::size_t>(first));
  const uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
  std::memcpy(out + first, nextSlot, static_cast<std::size_t>(4 - first));
}

inline void load8CachedFast(void *dst, const uint8_t *shadowBase,
                            uint64_t logicalOffset) {
  const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
  const uint8_t *slotBase =
      shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
  if (chunkOffset == 0) {
    std::memcpy(dst, slotBase, 8);
    return;
  }
  uint8_t *out = static_cast<uint8_t *>(dst);
  const uint64_t first = kLogicalGranuleBytes - chunkOffset;
  std::memcpy(out, slotBase + chunkOffset, static_cast<std::size_t>(first));
  const uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
  std::memcpy(out + first, nextSlot, static_cast<std::size_t>(8 - first));
}

inline void load16CachedFast(void *dst, const uint8_t *shadowBase,
                             uint64_t logicalOffset) {
  const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
  const uint8_t *slotBase =
      shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
  uint8_t *out = static_cast<uint8_t *>(dst);
  if (chunkOffset == 0) {
    std::memcpy(out, slotBase, 8);
    std::memcpy(out + 8, slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes), 8);
    return;
  }
  const uint64_t first = kLogicalGranuleBytes - chunkOffset;
  std::memcpy(out, slotBase + chunkOffset, static_cast<std::size_t>(first));
  const uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
  const uint64_t second = std::min<uint64_t>(16 - first, kLogicalGranuleBytes);
  std::memcpy(out + first, nextSlot, static_cast<std::size_t>(second));
  const uint64_t remaining = 16 - first - second;
  if (remaining != 0) {
    const uint8_t *thirdSlot = nextSlot + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
    std::memcpy(out + first + second, thirdSlot, static_cast<std::size_t>(remaining));
  }
}

void storeProtectedBytes(Range &range, void *dst, const void *src,
                         uint64_t size) {
  const auto *in = static_cast<const uint8_t *>(src);
  auto *logical = static_cast<uint8_t *>(dst);
  uint64_t logicalOffset = checkedLogicalOffset(range, dst, size, "store");
  const uint64_t counter =
      globalCounter().fetch_add(1, std::memory_order_relaxed);
  uint64_t remaining = size;
  while (remaining > 0) {
    const uint64_t chunkIndex = logicalOffset / kLogicalGranuleBytes;
    const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
    const uint64_t chunkBytes =
        std::min<uint64_t>(remaining, kLogicalGranuleBytes - chunkOffset);
    uint8_t *slotBase =
        range.shadow.data() +
        static_cast<std::ptrdiff_t>(chunkIndex * kShadowSlotBytes);
    std::memcpy(slotBase + chunkOffset, in, static_cast<std::size_t>(chunkBytes));
    std::memcpy(slotBase + kCounterOffset, &counter, sizeof(counter));
    in += chunkBytes;
    logicalOffset += chunkBytes;
    remaining -= chunkBytes;
  }
  if (mirrorRawStoreEnabled()) {
    std::memcpy(logical, src, static_cast<std::size_t>(size));
  }
}

void storeProtectedBytesCached(uint8_t *shadowBase, uint64_t logicalOffset,
                               const void *src, uint64_t size) {
  const auto *in = static_cast<const uint8_t *>(src);
  const uint64_t counter =
      globalCounter().fetch_add(1, std::memory_order_relaxed);
  uint64_t remaining = size;
  while (remaining > 0) {
    const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
    const uint64_t chunkBytes =
        std::min<uint64_t>(remaining, kLogicalGranuleBytes - chunkOffset);
    uint8_t *slotBase =
        shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
    std::memcpy(slotBase + chunkOffset, in, static_cast<std::size_t>(chunkBytes));
    std::memcpy(slotBase + kCounterOffset, &counter, sizeof(counter));
    in += chunkBytes;
    logicalOffset += chunkBytes;
    remaining -= chunkBytes;
  }
}

inline void storeTouchedCounter(uint8_t *slotBase, uint64_t counter) {
  std::memcpy(slotBase + kCounterOffset, &counter, sizeof(counter));
}

inline void store4CachedFast(uint8_t *shadowBase, uint64_t logicalOffset,
                             const void *src) {
  const auto *in = static_cast<const uint8_t *>(src);
  const uint64_t counter =
      globalCounter().fetch_add(1, std::memory_order_relaxed);
  const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
  uint8_t *slotBase =
      shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
  if (chunkOffset <= 4) {
    std::memcpy(slotBase + chunkOffset, in, 4);
    storeTouchedCounter(slotBase, counter);
    return;
  }
  const uint64_t first = kLogicalGranuleBytes - chunkOffset;
  std::memcpy(slotBase + chunkOffset, in, static_cast<std::size_t>(first));
  storeTouchedCounter(slotBase, counter);
  uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
  std::memcpy(nextSlot, in + first, static_cast<std::size_t>(4 - first));
  storeTouchedCounter(nextSlot, counter);
}

inline void store8CachedFast(uint8_t *shadowBase, uint64_t logicalOffset,
                             const void *src) {
  const auto *in = static_cast<const uint8_t *>(src);
  const uint64_t counter =
      globalCounter().fetch_add(1, std::memory_order_relaxed);
  const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
  uint8_t *slotBase =
      shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
  if (chunkOffset == 0) {
    std::memcpy(slotBase, in, 8);
    storeTouchedCounter(slotBase, counter);
    return;
  }
  const uint64_t first = kLogicalGranuleBytes - chunkOffset;
  std::memcpy(slotBase + chunkOffset, in, static_cast<std::size_t>(first));
  storeTouchedCounter(slotBase, counter);
  uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
  std::memcpy(nextSlot, in + first, static_cast<std::size_t>(8 - first));
  storeTouchedCounter(nextSlot, counter);
}

inline void store16CachedFast(uint8_t *shadowBase, uint64_t logicalOffset,
                              const void *src) {
  const auto *in = static_cast<const uint8_t *>(src);
  const uint64_t counter =
      globalCounter().fetch_add(1, std::memory_order_relaxed);
  const uint64_t chunkOffset = logicalOffset % kLogicalGranuleBytes;
  uint8_t *slotBase =
      shadowBase + static_cast<std::ptrdiff_t>(shadowByteOffset(logicalOffset) - chunkOffset);
  if (chunkOffset == 0) {
    std::memcpy(slotBase, in, 8);
    storeTouchedCounter(slotBase, counter);
    uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
    std::memcpy(nextSlot, in + 8, 8);
    storeTouchedCounter(nextSlot, counter);
    return;
  }
  const uint64_t first = kLogicalGranuleBytes - chunkOffset;
  std::memcpy(slotBase + chunkOffset, in, static_cast<std::size_t>(first));
  storeTouchedCounter(slotBase, counter);
  uint8_t *nextSlot = slotBase + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
  const uint64_t second = std::min<uint64_t>(16 - first, kLogicalGranuleBytes);
  std::memcpy(nextSlot, in + first, static_cast<std::size_t>(second));
  storeTouchedCounter(nextSlot, counter);
  const uint64_t remaining = 16 - first - second;
  if (remaining != 0) {
    uint8_t *thirdSlot = nextSlot + static_cast<std::ptrdiff_t>(kShadowSlotBytes);
    std::memcpy(thirdSlot, in + first + second, static_cast<std::size_t>(remaining));
    storeTouchedCounter(thirdSlot, counter);
  }
}

} // namespace

extern "C" uint64_t zebrafixExpandedSizeBytes(uint64_t logical_size) {
  return shadowSlotCount(logical_size) * kShadowSlotBytes;
}

extern "C" void zebrafixClearRanges() {
  ranges().clear();
  bumpRangesEpoch();
}

extern "C" void zebrafixRegisterRange(void *logical_base, uint64_t logical_size) {
  if (logical_base == nullptr || logical_size == 0) {
    return;
  }
  auto *base = static_cast<uint8_t *>(logical_base);
  auto *end = base + static_cast<std::ptrdiff_t>(logical_size);
  auto initializeShadow = [logical_base, logical_size](std::vector<uint8_t> &shadow) {
    const auto *bytes = static_cast<const uint8_t *>(logical_base);
    for (uint64_t logicalOffset = 0; logicalOffset < logical_size;
         logicalOffset += kLogicalGranuleBytes) {
      const uint64_t chunkBytes =
          std::min<uint64_t>(kLogicalGranuleBytes, logical_size - logicalOffset);
      uint8_t *slotBase =
          shadow.data() +
          static_cast<std::ptrdiff_t>(
              (logicalOffset / kLogicalGranuleBytes) * kShadowSlotBytes);
      std::memcpy(slotBase, bytes + logicalOffset, static_cast<std::size_t>(chunkBytes));
    }
  };
  auto &allRanges = ranges();
  allRanges.erase(
      std::remove_if(allRanges.begin(), allRanges.end(),
                     [base, end](const Range &range) {
                       return rangesOverlap(range, base, end);
                     }),
      allRanges.end());
  Range range{
      base,
      end,
      logical_size,
      std::vector<uint8_t>(
          static_cast<std::size_t>(zebrafixExpandedSizeBytes(logical_size)), 0),
  };
  initializeShadow(range.shadow);
  allRanges.push_back(std::move(range));
  sortRangesByBase();
  bumpRangesEpoch();
}

extern "C" void zebrafixUnregisterRange(void *logical_base) {
  auto &allRanges = ranges();
  allRanges.erase(
      std::remove_if(allRanges.begin(), allRanges.end(),
                     [logical_base](const Range &range) {
                       return range.base == logical_base;
                     }),
      allRanges.end());
  bumpRangesEpoch();
}

extern "C" void zebrafixLoadBytes(void *dst, const void *logical_addr,
                                  uint64_t size) {
  if (size == 0) {
    return;
  }
  if (const Range *range = findRangeForSpan(logical_addr, size)) {
    loadProtectedBytes(dst, *range, logical_addr, size);
    return;
  }
  std::memcpy(dst, logical_addr, static_cast<std::size_t>(size));
}

extern "C" void zebrafixStoreBytes(void *logical_addr, const void *src,
                                   uint64_t size) {
  if (size == 0) {
    return;
  }
  if (Range *range = findMutableRangeForSpan(logical_addr, size)) {
    storeProtectedBytes(*range, logical_addr, src, size);
    return;
  }
  std::memcpy(logical_addr, src, static_cast<std::size_t>(size));
}

extern "C" void *zebrafixLookupShadowBase(const void *logical_base) {
  if (logical_base == nullptr) {
    return nullptr;
  }
  Range *range = findRangeForAddress(logical_base);
  if (range == nullptr) {
    std::cerr << "zebrafix cached lookup failed base=" << logical_base << "\n";
    std::abort();
  }
  return range->shadow.data();
}

extern "C" uint64_t zebrafixLookupLogicalOffset(const void *logical_base) {
  if (logical_base == nullptr) {
    return 0;
  }
  Range *range = findRangeForAddress(logical_base);
  if (range == nullptr) {
    std::cerr << "zebrafix cached logical offset lookup failed base="
              << logical_base << "\n";
    std::abort();
  }
  return checkedLogicalOffset(*range, logical_base, 0, "lookup");
}

extern "C" void zebrafixLoadBytesCached(void *dst, const void *shadow_base,
                                        uint64_t logical_offset,
                                        uint64_t size) {
  if (size == 0) {
    return;
  }
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load missing shadow base\n";
    std::abort();
  }
  loadProtectedBytesCached(dst, static_cast<const uint8_t *>(shadow_base),
                           logical_offset, size);
}

extern "C" void zebrafixLoad4Cached(void *dst, const void *shadow_base,
                                    uint64_t logical_offset) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load4 missing shadow base\n";
    std::abort();
  }
  load4CachedFast(dst, static_cast<const uint8_t *>(shadow_base), logical_offset);
}

extern "C" void zebrafixLoad8Cached(void *dst, const void *shadow_base,
                                    uint64_t logical_offset) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load8 missing shadow base\n";
    std::abort();
  }
  load8CachedFast(dst, static_cast<const uint8_t *>(shadow_base), logical_offset);
}

extern "C" void zebrafixLoad16Cached(void *dst, const void *shadow_base,
                                     uint64_t logical_offset) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load16 missing shadow base\n";
    std::abort();
  }
  load16CachedFast(dst, static_cast<const uint8_t *>(shadow_base), logical_offset);
}

extern "C" uint32_t zebrafixLoad32Maybe(const void *logical_addr) {
  uint32_t value = 0;
  if (const Range *range = findRangeForSpan(logical_addr, sizeof(value))) {
    const uint64_t logicalOffset =
        checkedLogicalOffset(*range, logical_addr, sizeof(value), "load");
    load4CachedFast(&value, range->shadow.data(), logicalOffset);
    return value;
  }
  std::memcpy(&value, logical_addr, sizeof(value));
  return value;
}

extern "C" uint64_t zebrafixLoad64Maybe(const void *logical_addr) {
  uint64_t value = 0;
  if (const Range *range = findRangeForSpan(logical_addr, sizeof(value))) {
    const uint64_t logicalOffset =
        checkedLogicalOffset(*range, logical_addr, sizeof(value), "load");
    load8CachedFast(&value, range->shadow.data(), logicalOffset);
    return value;
  }
  std::memcpy(&value, logical_addr, sizeof(value));
  return value;
}

extern "C" unsigned __int128 zebrafixLoad128Maybe(const void *logical_addr) {
  unsigned __int128 value = 0;
  if (const Range *range = findRangeForSpan(logical_addr, sizeof(value))) {
    const uint64_t logicalOffset =
        checkedLogicalOffset(*range, logical_addr, sizeof(value), "load");
    load16CachedFast(&value, range->shadow.data(), logicalOffset);
    return value;
  }
  std::memcpy(&value, logical_addr, sizeof(value));
  return value;
}

extern "C" uint32_t zebrafixLoad32Value(const void *shadow_base,
                                        uint64_t logical_offset) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load32 missing shadow base\n";
    std::abort();
  }
  uint32_t value = 0;
  load4CachedFast(&value, static_cast<const uint8_t *>(shadow_base),
                  logical_offset);
  return value;
}

extern "C" uint64_t zebrafixLoad64Value(const void *shadow_base,
                                        uint64_t logical_offset) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load64 missing shadow base\n";
    std::abort();
  }
  uint64_t value = 0;
  load8CachedFast(&value, static_cast<const uint8_t *>(shadow_base),
                  logical_offset);
  return value;
}

extern "C" unsigned __int128 zebrafixLoad128Value(const void *shadow_base,
                                                  uint64_t logical_offset) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached load128 missing shadow base\n";
    std::abort();
  }
  unsigned __int128 value = 0;
  load16CachedFast(&value, static_cast<const uint8_t *>(shadow_base),
                   logical_offset);
  return value;
}

extern "C" void zebrafixStoreBytesCached(void *shadow_base,
                                         uint64_t logical_offset,
                                         const void *src, uint64_t size) {
  if (size == 0) {
    return;
  }
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store missing shadow base\n";
    std::abort();
  }
  storeProtectedBytesCached(static_cast<uint8_t *>(shadow_base), logical_offset,
                            src, size);
}

extern "C" void zebrafixStore4Cached(void *shadow_base, uint64_t logical_offset,
                                     const void *src) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store4 missing shadow base\n";
    std::abort();
  }
  store4CachedFast(static_cast<uint8_t *>(shadow_base), logical_offset, src);
}

extern "C" void zebrafixStore8Cached(void *shadow_base, uint64_t logical_offset,
                                     const void *src) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store8 missing shadow base\n";
    std::abort();
  }
  store8CachedFast(static_cast<uint8_t *>(shadow_base), logical_offset, src);
}

extern "C" void zebrafixStore16Cached(void *shadow_base, uint64_t logical_offset,
                                      const void *src) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store16 missing shadow base\n";
    std::abort();
  }
  store16CachedFast(static_cast<uint8_t *>(shadow_base), logical_offset, src);
}

extern "C" void zebrafixStore32Maybe(void *logical_addr, uint32_t value) {
  if (Range *range = findMutableRangeForSpan(logical_addr, sizeof(value))) {
    const uint64_t logicalOffset =
        checkedLogicalOffset(*range, logical_addr, sizeof(value), "store");
    store4CachedFast(range->shadow.data(), logicalOffset, &value);
    if (mirrorRawStoreEnabled()) {
      std::memcpy(logical_addr, &value, sizeof(value));
    }
    return;
  }
  std::memcpy(logical_addr, &value, sizeof(value));
}

extern "C" void zebrafixStore64Maybe(void *logical_addr, uint64_t value) {
  if (Range *range = findMutableRangeForSpan(logical_addr, sizeof(value))) {
    const uint64_t logicalOffset =
        checkedLogicalOffset(*range, logical_addr, sizeof(value), "store");
    store8CachedFast(range->shadow.data(), logicalOffset, &value);
    if (mirrorRawStoreEnabled()) {
      std::memcpy(logical_addr, &value, sizeof(value));
    }
    return;
  }
  std::memcpy(logical_addr, &value, sizeof(value));
}

extern "C" void zebrafixStore128Maybe(void *logical_addr,
                                      unsigned __int128 value) {
  if (Range *range = findMutableRangeForSpan(logical_addr, sizeof(value))) {
    const uint64_t logicalOffset =
        checkedLogicalOffset(*range, logical_addr, sizeof(value), "store");
    store16CachedFast(range->shadow.data(), logicalOffset, &value);
    if (mirrorRawStoreEnabled()) {
      std::memcpy(logical_addr, &value, sizeof(value));
    }
    return;
  }
  std::memcpy(logical_addr, &value, sizeof(value));
}

extern "C" void zebrafixStore32Value(void *shadow_base, uint64_t logical_offset,
                                     uint32_t value) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store32 missing shadow base\n";
    std::abort();
  }
  store4CachedFast(static_cast<uint8_t *>(shadow_base), logical_offset, &value);
}

extern "C" void zebrafixStore64Value(void *shadow_base, uint64_t logical_offset,
                                     uint64_t value) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store64 missing shadow base\n";
    std::abort();
  }
  store8CachedFast(static_cast<uint8_t *>(shadow_base), logical_offset, &value);
}

extern "C" void zebrafixStore128Value(void *shadow_base, uint64_t logical_offset,
                                      unsigned __int128 value) {
  if (shadow_base == nullptr) {
    std::cerr << "zebrafix cached store128 missing shadow base\n";
    std::abort();
  }
  store16CachedFast(static_cast<uint8_t *>(shadow_base), logical_offset, &value);
}

extern "C" void zebrafixMemcpy(void *dst, const void *src, uint64_t size) {
  if (size == 0) {
    return;
  }

  const Range *srcRange = findRangeForSpan(src, size);
  Range *dstRange = findMutableRangeForSpan(dst, size);

  if (srcRange == nullptr && dstRange == nullptr) {
    std::memcpy(dst, src, static_cast<std::size_t>(size));
    return;
  }
  if (srcRange == nullptr && dstRange != nullptr) {
    storeProtectedBytes(*dstRange, dst, src, size);
    return;
  }
  if (srcRange != nullptr && dstRange == nullptr) {
    loadProtectedBytes(dst, *srcRange, src, size);
    return;
  }

  std::vector<uint8_t> temp(static_cast<std::size_t>(size));
  loadProtectedBytes(temp.data(), *srcRange, src, size);
  storeProtectedBytes(*dstRange, dst, temp.data(), size);
}

extern "C" void zebrafixMemmove(void *dst, const void *src, uint64_t size) {
  if (size == 0 || dst == src) {
    return;
  }

  const Range *srcRange = findRangeForSpan(src, size);
  Range *dstRange = findMutableRangeForSpan(dst, size);

  if (srcRange == nullptr && dstRange == nullptr) {
    std::memmove(dst, src, static_cast<std::size_t>(size));
    return;
  }
  if (srcRange == nullptr && dstRange != nullptr) {
    storeProtectedBytes(*dstRange, dst, src, size);
    return;
  }
  if (srcRange != nullptr && dstRange == nullptr) {
    loadProtectedBytes(dst, *srcRange, src, size);
    return;
  }

  std::vector<uint8_t> temp(static_cast<std::size_t>(size));
  loadProtectedBytes(temp.data(), *srcRange, src, size);
  storeProtectedBytes(*dstRange, dst, temp.data(), size);
}

extern "C" void zebrafixMemset(void *dst, uint8_t value, uint64_t size) {
  if (size == 0) {
    return;
  }

  if (Range *range = findMutableRangeForSpan(dst, size)) {
    std::vector<uint8_t> temp(static_cast<std::size_t>(size), value);
    storeProtectedBytes(*range, dst, temp.data(), size);
    return;
  }
  std::memset(dst, value, static_cast<std::size_t>(size));
}
