#pragma once

#include "pin.H"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace pintrace {

constexpr uint32_t kTraceVersion = 4;
constexpr uint32_t kBlockSize = 16;

#pragma pack(push, 1)
struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t reserved;
  uint64_t page_size;
};

struct RecordHeader {
  uint64_t seq;
  uint32_t tid;
  uint32_t size;
  uint32_t size_read;
  uint32_t block16_read;
  uint64_t compat_addr;
  uint64_t paddr;
  uint64_t paddr16;
  uint64_t ip;
  uint64_t instr_id;
  uint32_t flags;
  uint32_t stack_depth;
};
#pragma pack(pop)

enum RecordFlags : uint32_t {
  kFlagPaddrValid = 1u << 0,
  kFlagPaddr16Valid = 1u << 1,
  kFlagUsedBefore = 1u << 2,
  kFlagPartialWriteRead = 1u << 3,
  kFlagPartialBlock16Read = 1u << 4,
  kFlagInitValue = 1u << 5,
  kFlagFirstTaintBefore = 1u << 6,
};

struct SiteMeta {
  uint64_t ip = 0;
  std::string ip_hex;
  std::string image;
  std::string offset;
  std::string routine;
  std::string symbol;
  std::string disasm;
};

inline std::string Hex(uint64_t value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%016" PRIx64, value);
  return std::string(buf);
}

inline std::string Offset(uint64_t value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "+0x%" PRIx64, value);
  return std::string(buf);
}

inline uint64_t AlignDown16(uint64_t value) {
  return value & ~static_cast<uint64_t>(kBlockSize - 1);
}

inline bool ParseU64(const std::string &text, uint64_t *out) {
  if (text.empty() || out == nullptr) {
    return false;
  }
  char *end = nullptr;
  const uint64_t value = std::strtoull(text.c_str(), &end, 0);
  if (end == text.c_str() || *end != '\0') {
    return false;
  }
  *out = value;
  return true;
}

inline std::string BaseName(const std::string &path) {
  const size_t pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

inline bool PathMatches(const std::string &path, const std::string &target) {
  return !target.empty() && (path == target || BaseName(path) == BaseName(target));
}

inline std::string JsonEscape(const std::string &text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(ch));
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  return out;
}

inline size_t SafeCopyZero(void *dst, ADDRINT src, size_t size) {
  const size_t copied = PIN_SafeCopy(dst, reinterpret_cast<const VOID *>(src), size);
  if (copied < size) {
    std::memset(static_cast<uint8_t *>(dst) + copied, 0, size - copied);
  }
  return copied;
}

inline SiteMeta DescribeInstruction(INS ins) {
  const uint64_t ip = static_cast<uint64_t>(INS_Address(ins));
  SiteMeta meta;
  meta.ip = ip;
  meta.ip_hex = Hex(ip);
  meta.disasm = INS_Disassemble(ins);

  const IMG img = IMG_FindByAddress(ip);
  if (IMG_Valid(img)) {
    meta.image = IMG_Name(img);
    meta.offset = Offset(ip - static_cast<uint64_t>(IMG_LowAddress(img)));
  }

  const RTN rtn = RTN_FindByAddress(ip);
  if (RTN_Valid(rtn)) {
    meta.routine = RTN_Name(rtn);
    meta.symbol = meta.routine;
    if (!meta.offset.empty()) {
      meta.symbol += meta.offset;
    }
  } else {
    meta.routine = "";
    meta.symbol = meta.offset;
  }
  return meta;
}

inline void RememberInstruction(std::map<uint64_t, SiteMeta> *ipmap, INS ins) {
  if (ipmap == nullptr) {
    return;
  }
  const uint64_t ip = static_cast<uint64_t>(INS_Address(ins));
  if (ipmap->find(ip) == ipmap->end()) {
    (*ipmap)[ip] = DescribeInstruction(ins);
  }
}

inline void WriteIpMap(const std::string &path, const std::map<uint64_t, SiteMeta> &ipmap) {
  if (path.empty()) {
    return;
  }
  std::ofstream out(path.c_str());
  if (!out) {
    return;
  }
  out << "# ip\timage\timage_offset\troutine\tdisasm\n";
  for (const auto &item : ipmap) {
    const SiteMeta &meta = item.second;
    out << meta.ip_hex << '\t' << meta.image << '\t' << meta.offset << '\t'
        << (meta.symbol.empty() ? meta.routine : meta.symbol) << '\t'
        << meta.disasm << '\n';
  }
}

inline void WriteFileHeader(FILE *out, uint64_t page_size) {
  FileHeader header{};
  std::memcpy(header.magic, "PADDRTRC", 8);
  header.version = kTraceVersion;
  header.reserved = 0;
  header.page_size = page_size;
  std::fwrite(&header, sizeof(header), 1, out);
}

}  // namespace pintrace
