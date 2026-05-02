#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

#include "cipherfix_taint_api.h"

namespace benchmark_runtime {

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

inline std::size_t pageSize() {
  const long value = ::sysconf(_SC_PAGESIZE);
  return value > 0 ? static_cast<std::size_t>(value) : 4096u;
}

inline std::size_t alignUp(std::size_t value, std::size_t alignment) {
  const std::size_t rem = value % alignment;
  return rem == 0 ? value : value + (alignment - rem);
}

inline bool fixedArenaEnabled() {
  const char *value = std::getenv("BUNDLE_FIXED_ARENA");
  if (!value) {
    value = std::getenv("GLOW_BUNDLE_FIXED_ARENA");
  }
  return !value || std::strcmp(value, "0") != 0;
}

inline std::uintptr_t fixedArenaBase() {
  const char *value = std::getenv("BUNDLE_FIXED_ARENA_BASE");
  if (!value) {
    value = std::getenv("GLOW_BUNDLE_FIXED_ARENA_BASE");
  }
  return value ? static_cast<std::uintptr_t>(std::strtoull(value, nullptr, 0))
               : 0x400000000000ULL;
}

inline void *heapAlloc(std::size_t alignment, std::size_t size) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, std::max<std::size_t>(alignment, sizeof(void *)), size) != 0 || !ptr) {
    std::cerr << "failed to allocate arena\n";
    std::exit(1);
  }
  std::memset(ptr, 0, size);
  cipherfixEnsureShadowForRange(ptr, size);
  return ptr;
}

class FixedAddressArena {
public:
  FixedAddressArena(std::size_t alignment, std::size_t constantSize,
                    std::size_t mutableSize, std::size_t activationSize)
      : constantSize_(constantSize), mutableSize_(mutableSize),
        activationSize_(activationSize) {
    fixedArenaEnabled() ? allocateFixed(alignment) : allocateHeap(alignment);
  }

  FixedAddressArena(const FixedAddressArena &) = delete;
  FixedAddressArena &operator=(const FixedAddressArena &) = delete;
  ~FixedAddressArena() { release(); }

  std::uint8_t *constantWeight() const { return constantWeight_; }
  std::uint8_t *mutableWeight() const { return mutableWeight_; }
  std::uint8_t *activations() const { return activations_; }

private:
  void allocateFixed(std::size_t alignment) {
    const std::size_t page = pageSize();
    const std::size_t regionAlign = std::max(alignment, page);
    constantOffset_ = 0;
    mutableOffset_ = alignUp(constantSize_, regionAlign);
    activationOffset_ = alignUp(mutableOffset_ + mutableSize_, regionAlign);
    mappingSize_ = alignUp(activationOffset_ + activationSize_, page);
    mapping_ = ::mmap(reinterpret_cast<void *>(fixedArenaBase()), mappingSize_,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapping_ == MAP_FAILED) {
      std::cerr << "failed to mmap fixed arena: errno=" << errno << " ("
                << std::strerror(errno) << ")\n";
      std::exit(1);
    }
    auto *base = static_cast<std::uint8_t *>(mapping_);
    constantWeight_ = base + constantOffset_;
    mutableWeight_ = base + mutableOffset_;
    activations_ = base + activationOffset_;
    cipherfixEnsureShadowForRange(mapping_, mappingSize_);
  }

  void allocateHeap(std::size_t alignment) {
    constantWeight_ = static_cast<std::uint8_t *>(heapAlloc(alignment, constantSize_));
    mutableWeight_ = static_cast<std::uint8_t *>(heapAlloc(alignment, mutableSize_));
    activations_ = static_cast<std::uint8_t *>(heapAlloc(alignment, activationSize_));
  }

  void release() {
    if (mapping_) {
      ::munmap(mapping_, mappingSize_);
      return;
    }
    std::free(constantWeight_);
    std::free(mutableWeight_);
    std::free(activations_);
  }

  std::size_t constantSize_ = 0;
  std::size_t mutableSize_ = 0;
  std::size_t activationSize_ = 0;
  std::size_t constantOffset_ = 0;
  std::size_t mutableOffset_ = 0;
  std::size_t activationOffset_ = 0;
  std::size_t mappingSize_ = 0;
  void *mapping_ = nullptr;
  std::uint8_t *constantWeight_ = nullptr;
  std::uint8_t *mutableWeight_ = nullptr;
  std::uint8_t *activations_ = nullptr;
};

} // namespace benchmark_runtime
