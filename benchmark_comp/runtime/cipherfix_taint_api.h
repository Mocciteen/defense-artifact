#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if defined(CIPHERFIX_TAINT_INPUT)
extern "C" {
void __attribute__((noinline, optimize("O0"))) classify(void *ptr, int length) {
  asm("");
}

void __attribute__((noinline, optimize("O0"))) declassify(void *ptr,
                                                           int length) {
  asm("");
}

void __attribute__((noinline, optimize("O0"))) drop_taint(void) { asm(""); }
}

inline constexpr std::intptr_t kCipherfixSecrecyBufferOffset = -0x2ffff000;
inline constexpr std::intptr_t kCipherfixMaskBufferOffset = -0x3ffff000;

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

inline int cipherfixLengthOrClamp(std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(size);
}

inline std::size_t cipherfixPageSize() {
  const long value = ::sysconf(_SC_PAGESIZE);
  return value > 0 ? static_cast<std::size_t>(value) : 4096u;
}

inline std::uintptr_t cipherfixAlignDown(std::uintptr_t value,
                                         std::size_t alignment) {
  return value & ~(static_cast<std::uintptr_t>(alignment) - 1u);
}

inline std::uintptr_t cipherfixAlignUp(std::uintptr_t value,
                                       std::size_t alignment) {
  const auto mask = static_cast<std::uintptr_t>(alignment) - 1u;
  return (value + mask) & ~mask;
}

inline void cipherfixClassifyInput(void *ptr, std::size_t size) {
  if (ptr != nullptr && size != 0) {
    classify(ptr, cipherfixLengthOrClamp(size));
  }
}

inline std::uint8_t *cipherfixSecrecyBytes(void *ptr) {
  const auto dataBase = reinterpret_cast<std::uintptr_t>(ptr);
  const auto secrecyBase = static_cast<std::uintptr_t>(
      static_cast<std::intptr_t>(dataBase) + kCipherfixSecrecyBufferOffset);
  return reinterpret_cast<std::uint8_t *>(secrecyBase);
}

inline std::uint8_t *cipherfixMaskBytes(void *ptr) {
  const auto dataBase = reinterpret_cast<std::uintptr_t>(ptr);
  const auto maskBase = static_cast<std::uintptr_t>(
      static_cast<std::intptr_t>(dataBase) + kCipherfixMaskBufferOffset);
  return reinterpret_cast<std::uint8_t *>(maskBase);
}

inline bool cipherfixRangeMapped(const void *ptr, std::size_t size) {
  if (ptr == nullptr || size == 0) {
    return false;
  }

  const std::size_t pageSize = cipherfixPageSize();
  const auto start = cipherfixAlignDown(reinterpret_cast<std::uintptr_t>(ptr), pageSize);
  const auto end = cipherfixAlignUp(reinterpret_cast<std::uintptr_t>(ptr) + size, pageSize);
  unsigned char vec = 0;
  for (auto page = start; page < end; page += pageSize) {
    if (::mincore(reinterpret_cast<void *>(page), pageSize, &vec) != 0) {
      return false;
    }
  }
  return true;
}

inline void cipherfixEnsureMappedPages(std::uintptr_t start, std::uintptr_t end,
                                       const char *label) {
  const std::size_t pageSize = cipherfixPageSize();
  unsigned char vec = 0;
  for (auto page = start; page < end; page += pageSize) {
    if (::mincore(reinterpret_cast<void *>(page), pageSize, &vec) == 0) {
      continue;
    }

    if (errno != ENOMEM) {
      const int savedErrno = errno;
      std::cerr << "cipherfix: mincore failed for " << label << " page 0x"
                << std::hex << page << std::dec << " errno=" << savedErrno
                << " (" << std::strerror(savedErrno) << ")\n";
      std::exit(1);
    }

    void *mapped = ::mmap(reinterpret_cast<void *>(page), pageSize,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                          -1, 0);
    if (mapped == MAP_FAILED && errno != EEXIST) {
      const int savedErrno = errno;
      std::cerr << "cipherfix: mmap failed for " << label << " page 0x"
                << std::hex << page << std::dec << " errno=" << savedErrno
                << " (" << std::strerror(savedErrno) << ")\n";
      std::exit(1);
    }
  }
}

inline void cipherfixEnsureShadowForRange(void *ptr, std::size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const std::size_t pageSize = cipherfixPageSize();
  const auto secrecyStart = cipherfixAlignDown(
      reinterpret_cast<std::uintptr_t>(cipherfixSecrecyBytes(ptr)), pageSize);
  const auto secrecyEnd = cipherfixAlignUp(
      reinterpret_cast<std::uintptr_t>(cipherfixSecrecyBytes(ptr)) + size,
      pageSize);
  const auto maskStart = cipherfixAlignDown(
      reinterpret_cast<std::uintptr_t>(cipherfixMaskBytes(ptr)), pageSize);
  const auto maskEnd = cipherfixAlignUp(
      reinterpret_cast<std::uintptr_t>(cipherfixMaskBytes(ptr)) + size, pageSize);

  cipherfixEnsureMappedPages(secrecyStart, secrecyEnd, "secrecy");
  cipherfixEnsureMappedPages(maskStart, maskEnd, "mask");
}

inline bool cipherfixShadowReady(void *ptr, std::size_t size) {
  if (ptr == nullptr || size == 0) {
    return false;
  }
  return cipherfixRangeMapped(cipherfixSecrecyBytes(ptr), size) &&
         cipherfixRangeMapped(cipherfixMaskBytes(ptr), size);
}

inline void cipherfixMarkSecretRange(void *ptr, std::size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  auto *secrecy = cipherfixSecrecyBytes(ptr);
  std::fill_n(secrecy, size, static_cast<std::uint8_t>(0xff));
}

inline void cipherfixApplySecretRangesFile(const char *path, void *regionPtr,
                                           std::size_t regionSize) {
  if (path == nullptr || path[0] == '\0' || regionPtr == nullptr ||
      regionSize == 0) {
    return;
  }

  std::ifstream input(path);
  if (!input) {
    return;
  }

  const auto regionStart = reinterpret_cast<std::uintptr_t>(regionPtr);
  const auto regionEnd = regionStart + regionSize;
  auto *secrecy = cipherfixSecrecyBytes(regionPtr);

  std::string startText;
  std::string endText;
  while (input >> startText >> endText) {
    const auto start = static_cast<std::uintptr_t>(
        std::strtoull(startText.c_str(), nullptr, 0));
    const auto end = static_cast<std::uintptr_t>(
        std::strtoull(endText.c_str(), nullptr, 0));
    if (end <= start || end <= regionStart || regionEnd <= start) {
      continue;
    }

    const auto clippedStart = std::max(start, regionStart);
    const auto clippedEnd = std::min(end, regionEnd);
    const auto offset = clippedStart - regionStart;
    const auto length = clippedEnd - clippedStart;
    std::fill_n(secrecy + offset, length, static_cast<std::uint8_t>(0xff));
  }
}

inline void cipherfixDecodeAndClearShadow(void *ptr, std::size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  auto *data = reinterpret_cast<std::uint8_t *>(ptr);
  auto *secrecy = cipherfixSecrecyBytes(ptr);
  auto *mask = cipherfixMaskBytes(ptr);

  for (std::size_t index = 0; index < size; ++index) {
    data[index] ^= static_cast<std::uint8_t>(mask[index] & secrecy[index]);
    mask[index] = 0;
    secrecy[index] = 0;
  }
}

inline void cipherfixDeclassifyBuffer(void *ptr, std::size_t size) {
  if (ptr != nullptr && size != 0) {
    declassify(ptr, cipherfixLengthOrClamp(size));
    if (cipherfixShadowReady(ptr, size)) {
      cipherfixDecodeAndClearShadow(ptr, size);
    }
  }
}

inline void cipherfixDropTaintState(void) { drop_taint(); }
#else
inline void cipherfixClassifyInput(void *, std::size_t) {}
inline void cipherfixMarkSecretRange(void *, std::size_t) {}
inline void cipherfixApplySecretRangesFile(const char *, void *, std::size_t) {}
inline void cipherfixEnsureShadowForRange(void *, std::size_t) {}
inline void cipherfixDeclassifyBuffer(void *, std::size_t) {}
inline void cipherfixDropTaintState(void) {}
#endif
