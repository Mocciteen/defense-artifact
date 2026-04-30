#include "pin.H"

#include <unistd.h>
#include <sys/syscall.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

static KNOB<std::string> KnobOutput(
    KNOB_MODE_WRITEONCE, "pintool", "o", "taint_block16_bits.json",
    "Output JSON containing per-tainted-16B-block bitstreams");
static KNOB<std::string> KnobIpMap(
    KNOB_MODE_WRITEONCE, "pintool", "m", "",
    "Optional IP map output. Written as text for easier disassembly lookup.");
static KNOB<std::string> KnobSiteIp(
    KNOB_MODE_WRITEONCE, "pintool", "site-ip", "",
    "Optional exact write-site IP filter. Only matching site writes are kept.");
static KNOB<std::string> KnobCallerIp(
    KNOB_MODE_WRITEONCE, "pintool", "caller-ip", "",
    "Optional exact immediate caller callsite IP filter. Only writes whose "
    "current callstack top matches this callsite are kept.");
static KNOB<UINT32> KnobStackDepth(
    KNOB_MODE_WRITEONCE, "pintool", "stack-depth", "0",
    "Ignored compatibility knob.");
static KNOB<BOOL> KnobNoPaddr(
    KNOB_MODE_WRITEONCE, "pintool", "no-paddr", "1",
    "Ignored compatibility knob.");
static KNOB<std::string> KnobTaintFile(
    KNOB_MODE_WRITEONCE, "pintool", "taint-file", "",
    "Seed taint from bytes read() from this file (path or basename). "
    "Empty disables file-based seeding.");
static KNOB<std::string> KnobTaintSeedMode(
    KNOB_MODE_WRITEONCE, "pintool", "taint-seed-mode", "file",
    "Taint seed mode: file|input-tensor.");
static KNOB<BOOL> KnobTaintNoLock(
    KNOB_MODE_WRITEONCE, "pintool", "taint-no-lock", "0",
    "Disable locks for taint shadow state (unsafe if multiple threads, "
    "faster).");
static KNOB<BOOL> KnobTaintOnly(
    KNOB_MODE_WRITEONCE, "pintool", "taint-only", "1",
    "Ignored compatibility knob. This tool is always taint-filtered.");
static KNOB<std::string> KnobTaintDecimalOut(
    KNOB_MODE_WRITEONCE, "pintool", "taint-decimal-out", "",
    "Ignored compatibility knob.");
static KNOB<std::string> KnobTaintDecimalFmt(
    KNOB_MODE_WRITEONCE, "pintool", "taint-decimal-fmt", "auto",
    "Ignored compatibility knob.");
static KNOB<std::string> KnobUnchangedEventBin(
    KNOB_MODE_WRITEONCE, "pintool", "unchanged-event-bin", "",
    "Optional binary output for exact unchanged block-overlap events.");
static KNOB<std::string> KnobBlockEventBin(
    KNOB_MODE_WRITEONCE, "pintool", "block-event-bin", "",
    "Optional binary output for every taint-filtered block-overlap compare.");
static KNOB<std::string> KnobReadEventBin(
    KNOB_MODE_WRITEONCE, "pintool", "read-event-bin", "",
    "Optional binary output for selected memory-read events.");
static KNOB<std::string> KnobReadSiteIpList(
    KNOB_MODE_WRITEONCE, "pintool", "read-site-ip-list", "",
    "Comma-separated exact read-site IP filter for -read-event-bin.");
static KNOB<BOOL> KnobAllWrites(
    KNOB_MODE_WRITEONCE, "pintool", "all-writes", "0",
    "If 1, bypass taint gating and record every overlapping write for the "
    "selected site/caller.");
static KNOB<BOOL> KnobSiteBitsOnly(
    KNOB_MODE_WRITEONCE, "pintool", "site-bits-only", "0",
    "If 1, require -site-ip and emit only the matching site's 01 string. "
    "Per-address aggregation and block/unchanged event streams are skipped.");
static KNOB<std::string> KnobSiteBitsText(
    KNOB_MODE_WRITEONCE, "pintool", "site-bits-txt", "",
    "Optional raw 01 output for the selected site's bit string when "
    "-site-bits-only 1.");

static PIN_LOCK g_stream_lock;
static PIN_LOCK g_taint_lock;
static TLS_KEY g_tls_key = INVALID_TLS_KEY;

static bool g_taint_enabled = false;
static bool g_taint_no_lock = false;
static bool g_seed_from_input_tensor = false;
static std::string g_taint_file;
static std::string g_taint_file_base;
static std::unordered_set<int> g_taint_fds;
static bool g_has_site_ip_filter = false;
static uint64_t g_site_ip_filter = 0;
static bool g_has_caller_ip_filter = false;
static uint64_t g_caller_ip_filter = 0;
static bool g_has_read_site_ip_filters = false;
static std::unordered_set<uint64_t> g_read_site_ip_filters;
static bool g_record_all_writes = false;
static bool g_site_bits_only = false;

constexpr size_t kBlockSize = 16;
constexpr size_t kMaxSavedMemOps = 8;

struct SiteMeta {
  uint64_t ip = 0;
  std::string ipHex;
  std::string image;
  std::string imageOffset;
  std::string routine;
  std::string site;
  std::string disasm;
};

struct SavedBlock {
  uint64_t blockAddr = 0;
  std::array<uint8_t, kBlockSize> before{};
};

struct SavedWriteSlot {
  bool valid = false;
  uint64_t writeAddr = 0;
  uint32_t writeSize = 0;
  std::vector<SavedBlock> blocks;
};

struct ThreadData {
  std::array<SavedWriteSlot, kMaxSavedMemOps> saved;
  std::array<uint8_t, kBlockSize> after{};
  std::vector<uint64_t> callstack;

  bool cur_taint = false;
  std::vector<uint8_t> reg_taint;

  uint32_t update_inputs_depth = 0;
  uint32_t tensor_assign_depth = 0;

  ADDRINT last_sys_num = 0;
  ADDRINT last_sys_arg0 = 0;
  ADDRINT last_sys_arg1 = 0;
  ADDRINT last_sys_arg2 = 0;
  ADDRINT last_sys_arg3 = 0;
  ADDRINT last_sys_arg4 = 0;
  bool last_open_match = false;
};

struct ShadowPage {
  std::array<uint64_t, 64> bits{};
};

struct BlockState {
  uint64_t vaddr = 0;
  uint64_t writeCount = 0;
  uint64_t unchangedCount = 0;
  uint64_t changedCount = 0;
  bool hasInitialBaseline = false;
  std::string bits;
  std::map<uint64_t, uint64_t> ownerCounts;
};

struct SiteState {
  uint64_t ip = 0;
  uint64_t writeCount = 0;
  uint64_t unchangedCount = 0;
  uint64_t changedCount = 0;
  std::string bits;
};

struct UnchangedEventFileHeader {
  char magic[8];
  uint32_t version = 0;
  uint32_t recordSize = 0;
  uint64_t blockSize = 0;
  uint64_t reserved = 0;
};

struct UnchangedEventRecord {
  uint64_t seq = 0;
  uint64_t blockVaddr = 0;
  uint64_t siteIp = 0;
  uint64_t writeVaddr = 0;
  uint64_t blockCompareIndex = 0;
  uint64_t blockUnchangedIndex = 0;
  uint32_t writeSize = 0;
  uint16_t overlapOffset = 0;
  uint16_t overlapSize = 0;
  std::array<uint8_t, kBlockSize> blockBefore{};
  std::array<uint8_t, kBlockSize> overlapValue{};
};

struct BlockEventFileHeader {
  char magic[8];
  uint32_t version = 0;
  uint32_t recordSize = 0;
  uint64_t blockSize = 0;
  uint64_t reserved = 0;
};

struct BlockEventRecord {
  uint64_t seq = 0;
  uint64_t blockVaddr = 0;
  uint64_t siteIp = 0;
  uint64_t writeVaddr = 0;
  uint64_t blockCompareIndex = 0;
  uint64_t blockUnchangedIndex = 0;
  uint32_t writeSize = 0;
  uint16_t overlapOffset = 0;
  uint16_t overlapSize = 0;
  uint64_t flags = 0;
  std::array<uint8_t, kBlockSize> blockBefore{};
  std::array<uint8_t, kBlockSize> blockAfter{};
};

struct ReadEventFileHeader {
  char magic[8];
  uint32_t version = 0;
  uint32_t recordSize = 0;
  uint64_t reserved0 = 0;
  uint64_t reserved1 = 0;
};

struct ReadEventRecord {
  uint64_t seq = 0;
  uint64_t siteIp = 0;
  uint64_t readVaddr = 0;
  uint64_t callerIp = 0;
  uint32_t readSize = 0;
  uint16_t memOp = 0;
  uint16_t flags = 0;
};

static_assert(sizeof(UnchangedEventFileHeader) == 32,
              "Unexpected unchanged event file header size");
static_assert(sizeof(UnchangedEventRecord) == 88,
              "Unexpected unchanged event record size");
static_assert(sizeof(BlockEventFileHeader) == 32,
              "Unexpected block event file header size");
static_assert(sizeof(BlockEventRecord) == 96,
              "Unexpected block event record size");
static_assert(sizeof(ReadEventFileHeader) == 32,
              "Unexpected read event file header size");
static_assert(sizeof(ReadEventRecord) == 40,
              "Unexpected read event record size");

static std::unordered_map<uint64_t, ShadowPage *> g_shadow_pages;
static std::map<uint64_t, SiteMeta> g_siteMetaByIp;
static std::map<uint64_t, BlockState> g_blocks;
static std::map<uint64_t, SiteState> g_sites;
static std::set<std::string> g_sitesWithoutAfter;
static std::set<uint64_t> g_siteBitsOnlyBlocks;
static std::ofstream g_unchangedEventBin;
static uint64_t g_unchangedEventCount = 0;
static std::ofstream g_blockEventBin;
static uint64_t g_blockEventCount = 0;
static std::ofstream g_readEventBin;
static uint64_t g_readEventCount = 0;

static uint64_t CurrentCallerIp(const ThreadData *td);

static std::string BaseName(const std::string &path) {
  const size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

static bool PathMatchesTaintFile(const std::string &path) {
  if (!g_taint_enabled || g_taint_file.empty()) {
    return false;
  }
  if (path == g_taint_file) {
    return true;
  }
  const std::string base = BaseName(path);
  return (!g_taint_file_base.empty() && base == g_taint_file_base);
}

static bool parseUint64Arg(const std::string &text, uint64_t *out) {
  if (out == nullptr || text.empty()) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 0);
  if (errno != 0 || end == text.c_str() || (end != nullptr && *end != '\0')) {
    return false;
  }
  *out = static_cast<uint64_t>(parsed);
  return true;
}

static bool parseUint64ListArg(const std::string &text,
                               std::unordered_set<uint64_t> *out) {
  if (out == nullptr) {
    return false;
  }
  out->clear();
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next = text.find(',', pos);
    if (next == std::string::npos) {
      next = text.size();
    }
    size_t begin = pos;
    while (begin < next &&
           std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
      begin++;
    }
    size_t end = next;
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
      end--;
    }
    if (begin < end) {
      uint64_t value = 0;
      if (!parseUint64Arg(text.substr(begin, end - begin), &value)) {
        return false;
      }
      out->insert(value);
    }
    pos = next + 1;
  }
  return !out->empty();
}

static std::string hex64(uint64_t value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%016" PRIx64, value);
  return std::string(buf);
}

static std::string hexAddr(uint64_t value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "0x%" PRIx64, value);
  return std::string(buf);
}

static std::string offsetString(uint64_t value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "+0x%" PRIx64, value);
  return std::string(buf);
}

static std::string jsonEscape(const std::string &input) {
  std::string out;
  out.reserve(input.size() + 16);
  for (const unsigned char ch : input) {
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
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
      if (ch < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
        out += buf;
      } else {
        out.push_back(static_cast<char>(ch));
      }
      break;
    }
  }
  return out;
}

static INT32 Usage() {
  std::fprintf(
      stderr,
      "taintblock16trace: emit per-tainted-16B-block 01 strings over writes\n"
      "  -o <file>                        output JSON "
      "(default: taint_block16_bits.json)\n"
      "  -block-event-bin <file>          optional exact binary stream for all "
      "block compares\n"
      "  -unchanged-event-bin <file>      optional exact unchanged-event "
      "binary stream\n"
      "  -read-event-bin <file>           optional exact binary stream for "
      "selected memory reads\n"
      "  -read-site-ip-list <a,b,...>     comma-separated exact read-site IP "
      "filter for -read-event-bin\n"
      "  -site-ip <addr>                  optional exact write-site IP "
      "filter\n"
      "  -caller-ip <addr>                optional exact immediate caller "
      "callsite IP filter\n"
      "  -all-writes 0|1                 bypass taint gating for selected "
      "site/caller\n"
      "  -site-bits-only 0|1             only keep the selected site's "
      "aggregated 01 string\n"
      "  -site-bits-txt <file>           optional raw 01 output for "
      "-site-bits-only\n"
      "  -taint-file <path>              seed taint from read() of this file\n"
      "  -taint-seed-mode file|input-tensor  taint seed mode\n"
      "  -taint-no-lock 0|1              disable taint shadow locks\n");
  return -1;
}

static VOID emitBlockEvent(const SavedWriteSlot &slot, const SavedBlock &block,
                           const SiteMeta &meta, uint64_t blockCompareIndex,
                           uint64_t blockUnchangedIndex, bool changed,
                           const std::array<uint8_t, kBlockSize> &after) {
  if (!g_blockEventBin.is_open() || slot.writeSize == 0) {
    return;
  }

  const uint64_t writeStart = slot.writeAddr;
  const uint64_t writeEnd = slot.writeAddr + static_cast<uint64_t>(slot.writeSize);
  const uint64_t blockStart = block.blockAddr;
  const uint64_t blockEnd = block.blockAddr + kBlockSize;
  const uint64_t overlapStart = std::max(writeStart, blockStart);
  const uint64_t overlapEnd = std::min(writeEnd, blockEnd);
  if (overlapStart >= overlapEnd) {
    return;
  }

  BlockEventRecord record;
  record.seq = g_blockEventCount + 1;
  record.blockVaddr = block.blockAddr;
  record.siteIp = meta.ip;
  record.writeVaddr = slot.writeAddr;
  record.blockCompareIndex = blockCompareIndex;
  record.blockUnchangedIndex = changed ? 0 : blockUnchangedIndex;
  record.writeSize = slot.writeSize;
  record.overlapOffset = static_cast<uint16_t>(overlapStart - blockStart);
  record.overlapSize = static_cast<uint16_t>(overlapEnd - overlapStart);
  record.flags = changed ? 1ULL : 0ULL;
  std::memcpy(record.blockBefore.data(), block.before.data(), kBlockSize);
  std::memcpy(record.blockAfter.data(), after.data(), kBlockSize);

  g_blockEventBin.write(reinterpret_cast<const char *>(&record), sizeof(record));
  if (g_blockEventBin) {
    g_blockEventCount += 1;
  }
}

static VOID emitUnchangedEvent(const SavedWriteSlot &slot,
                               const SavedBlock &block, const SiteMeta &meta,
                               uint64_t blockCompareIndex,
                               uint64_t blockUnchangedIndex,
                               const std::array<uint8_t, kBlockSize> &after) {
  if (!g_unchangedEventBin.is_open() || slot.writeSize == 0) {
    return;
  }

  const uint64_t writeStart = slot.writeAddr;
  const uint64_t writeEnd = slot.writeAddr + static_cast<uint64_t>(slot.writeSize);
  const uint64_t blockStart = block.blockAddr;
  const uint64_t blockEnd = block.blockAddr + kBlockSize;
  const uint64_t overlapStart = std::max(writeStart, blockStart);
  const uint64_t overlapEnd = std::min(writeEnd, blockEnd);
  if (overlapStart >= overlapEnd) {
    return;
  }

  UnchangedEventRecord record;
  record.seq = g_unchangedEventCount + 1;
  record.blockVaddr = block.blockAddr;
  record.siteIp = meta.ip;
  record.writeVaddr = slot.writeAddr;
  record.blockCompareIndex = blockCompareIndex;
  record.blockUnchangedIndex = blockUnchangedIndex;
  record.writeSize = slot.writeSize;
  record.overlapOffset = static_cast<uint16_t>(overlapStart - blockStart);
  record.overlapSize = static_cast<uint16_t>(overlapEnd - overlapStart);
  std::memcpy(record.blockBefore.data(), block.before.data(), kBlockSize);
  std::memcpy(record.overlapValue.data(), after.data() + record.overlapOffset,
              record.overlapSize);

  g_unchangedEventBin.write(reinterpret_cast<const char *>(&record),
                            sizeof(record));
  if (g_unchangedEventBin) {
    g_unchangedEventCount += 1;
  }
}

static VOID emitReadEvent(THREADID tid, UINT32 memOp, ADDRINT ea, UINT32 size,
                          ADDRINT ip) {
  if (!g_readEventBin.is_open() || size == 0) {
    return;
  }

  const uint64_t siteIp = static_cast<uint64_t>(ip);
  if (g_has_read_site_ip_filters &&
      g_read_site_ip_filters.find(siteIp) == g_read_site_ip_filters.end()) {
    return;
  }

  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  const uint64_t callerIp = CurrentCallerIp(td);
  if (g_has_caller_ip_filter && callerIp != g_caller_ip_filter) {
    return;
  }

  ReadEventRecord record;
  record.seq = g_readEventCount + 1;
  record.siteIp = siteIp;
  record.readVaddr = static_cast<uint64_t>(ea);
  record.callerIp = callerIp;
  record.readSize = size;
  record.memOp = static_cast<uint16_t>(memOp);
  record.flags = 0;

  PIN_GetLock(&g_stream_lock, tid + 1);
  g_readEventBin.write(reinterpret_cast<const char *>(&record), sizeof(record));
  if (g_readEventBin) {
    g_readEventCount += 1;
  }
  PIN_ReleaseLock(&g_stream_lock);
}

static SiteMeta describeSite(INS ins) {
  const uint64_t ip = static_cast<uint64_t>(INS_Address(ins));

  SiteMeta meta;
  meta.ip = ip;
  meta.ipHex = hex64(ip);
  meta.image = "<unknown>";
  meta.imageOffset = "+0x0";
  meta.routine = "-";
  meta.site = meta.ipHex;
  meta.disasm = INS_Disassemble(ins);

  const IMG img = IMG_FindByAddress(INS_Address(ins));
  if (IMG_Valid(img)) {
    meta.image = IMG_Name(img);
    meta.imageOffset =
        offsetString(ip - static_cast<uint64_t>(IMG_LowAddress(img)));
  }

  const RTN rtn = RTN_FindByAddress(INS_Address(ins));
  if (RTN_Valid(rtn)) {
    meta.routine = RTN_Name(rtn);
    meta.site =
        meta.routine + offsetString(ip - static_cast<uint64_t>(RTN_Address(rtn)));
  }

  return meta;
}

static SiteMeta defaultSiteMeta(uint64_t ip) {
  SiteMeta meta;
  meta.ip = ip;
  meta.ipHex = hex64(ip);
  meta.image = "<unknown>";
  meta.imageOffset = "+0x0";
  meta.routine = "-";
  meta.site = meta.ipHex;
  meta.disasm = "<unknown>";
  return meta;
}

static ShadowPage *GetShadowPage(uint64_t pageNo, bool create) {
  std::unordered_map<uint64_t, ShadowPage *>::iterator it =
      g_shadow_pages.find(pageNo);
  if (it != g_shadow_pages.end()) {
    return it->second;
  }
  if (!create) {
    return nullptr;
  }
  ShadowPage *page = new ShadowPage();
  g_shadow_pages.emplace(pageNo, page);
  return page;
}

static inline void ShadowSetByte(ShadowPage *page, uint32_t byteOff,
                                 bool tainted) {
  const uint32_t bit = byteOff & 63u;
  const uint32_t word = byteOff >> 6;
  const uint64_t mask = (1ULL << bit);
  if (tainted) {
    page->bits[word] |= mask;
  } else {
    page->bits[word] &= ~mask;
  }
}

static inline bool ShadowGetByte(const ShadowPage *page, uint32_t byteOff) {
  const uint32_t bit = byteOff & 63u;
  const uint32_t word = byteOff >> 6;
  return ((page->bits[word] >> bit) & 1ULL) != 0;
}

static bool MemAnyTaint(uint64_t addr, uint32_t size) {
  if (size == 0) {
    return false;
  }
  const uint64_t end = addr + static_cast<uint64_t>(size);
  for (uint64_t cur = addr; cur < end; ++cur) {
    const uint64_t pageNo = cur >> 12;
    const uint32_t byteOff = static_cast<uint32_t>(cur & 0xFFF);
    const ShadowPage *page = nullptr;
    if (!g_taint_no_lock) {
      PIN_GetLock(&g_taint_lock, 1);
    }
    std::unordered_map<uint64_t, ShadowPage *>::const_iterator it =
        g_shadow_pages.find(pageNo);
    if (it != g_shadow_pages.end()) {
      page = it->second;
    }
    if (!g_taint_no_lock) {
      PIN_ReleaseLock(&g_taint_lock);
    }
    if (page != nullptr && ShadowGetByte(page, byteOff)) {
      return true;
    }
  }
  return false;
}

static void MemSetTaint(uint64_t addr, uint32_t size, bool tainted) {
  if (size == 0) {
    return;
  }
  const uint64_t end = addr + static_cast<uint64_t>(size);
  for (uint64_t cur = addr; cur < end; ++cur) {
    const uint64_t pageNo = cur >> 12;
    const uint32_t byteOff = static_cast<uint32_t>(cur & 0xFFF);
    if (!g_taint_no_lock) {
      PIN_GetLock(&g_taint_lock, 1);
    }
    ShadowPage *page = GetShadowPage(pageNo, tainted);
    if (page != nullptr) {
      ShadowSetByte(page, byteOff, tainted);
    }
    if (!g_taint_no_lock) {
      PIN_ReleaseLock(&g_taint_lock);
    }
  }
}

static void MemSetTaintRange(uint64_t addr, uint64_t size, bool tainted) {
  while (size > 0) {
    const uint32_t chunk =
        size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? std::numeric_limits<uint32_t>::max()
            : static_cast<uint32_t>(size);
    MemSetTaint(addr, chunk, tainted);
    addr += chunk;
    size -= chunk;
  }
}

static bool IsTaintFd(int fd) {
  bool isTaintFd = false;
  if (!g_taint_no_lock) {
    PIN_GetLock(&g_taint_lock, 1);
  }
  isTaintFd = (g_taint_fds.find(fd) != g_taint_fds.end());
  if (!g_taint_no_lock) {
    PIN_ReleaseLock(&g_taint_lock);
  }
  return isTaintFd;
}

static inline REG NormReg(REG reg) {
  if (reg == REG_INVALID()) {
    return reg;
  }
  return REG_FullRegName(reg);
}

static inline bool GetRegTaint(ThreadData *td, REG reg) {
  reg = NormReg(reg);
  if (reg == REG_INVALID()) {
    return false;
  }
  const uint32_t idx = static_cast<uint32_t>(reg);
  if (idx >= td->reg_taint.size()) {
    return false;
  }
  return td->reg_taint[idx] != 0;
}

static inline void SetRegTaint(ThreadData *td, REG reg, bool tainted) {
  reg = NormReg(reg);
  if (reg == REG_INVALID()) {
    return;
  }
  const uint32_t idx = static_cast<uint32_t>(reg);
  if (idx >= td->reg_taint.size()) {
    return;
  }
  td->reg_taint[idx] = tainted ? 1 : 0;
}

static inline bool currentWriteIsSeeded(const ThreadData *td) {
  return g_seed_from_input_tensor &&
         (td->tensor_assign_depth > 0 || td->update_inputs_depth > 0);
}

static inline bool currentWriteIsTainted(const ThreadData *td) {
  return td->cur_taint || currentWriteIsSeeded(td);
}

static VOID TaintBegin(THREADID tid) {
  if (!g_taint_enabled) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  td->cur_taint = false;
}

static VOID TaintAccReg(THREADID tid, UINT32 regValue) {
  if (!g_taint_enabled) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  if (GetRegTaint(td, static_cast<REG>(regValue))) {
    td->cur_taint = true;
  }
}

static VOID TaintAccMem(THREADID tid, ADDRINT addr, UINT32 size) {
  if (!g_taint_enabled) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  if (MemAnyTaint(static_cast<uint64_t>(addr), size)) {
    td->cur_taint = true;
  }
}

static VOID TaintSetReg(THREADID tid, UINT32 regValue) {
  if (!g_taint_enabled) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  SetRegTaint(td, static_cast<REG>(regValue), td->cur_taint);
}

static VOID TaintSetMemSeedAware(THREADID tid, ADDRINT addr, UINT32 size) {
  if (!g_taint_enabled) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  MemSetTaint(static_cast<uint64_t>(addr), size, currentWriteIsTainted(td));
}

static VOID OnEnterUpdateInputs(THREADID tid) {
  if (!g_seed_from_input_tensor) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr) {
    td->update_inputs_depth++;
  }
}

static VOID OnExitUpdateInputs(THREADID tid) {
  if (!g_seed_from_input_tensor) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr && td->update_inputs_depth > 0) {
    td->update_inputs_depth--;
  }
}

static VOID OnEnterTensorAssign(THREADID tid) {
  if (!g_seed_from_input_tensor) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || td->update_inputs_depth == 0) {
    return;
  }
  td->tensor_assign_depth++;
}

static VOID OnExitTensorAssign(THREADID tid) {
  if (!g_seed_from_input_tensor) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr && td->tensor_assign_depth > 0) {
    td->tensor_assign_depth--;
  }
}

static VOID SyscallEntry(THREADID tid, CONTEXT *ctx, SYSCALL_STANDARD std,
                         VOID *) {
  if (!g_taint_enabled || g_seed_from_input_tensor) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  td->last_sys_num = PIN_GetSyscallNumber(ctx, std);
  td->last_sys_arg0 = PIN_GetSyscallArgument(ctx, std, 0);
  td->last_sys_arg1 = PIN_GetSyscallArgument(ctx, std, 1);
  td->last_sys_arg2 = PIN_GetSyscallArgument(ctx, std, 2);
  td->last_sys_arg3 = PIN_GetSyscallArgument(ctx, std, 3);
  td->last_sys_arg4 = PIN_GetSyscallArgument(ctx, std, 4);
  td->last_open_match = false;

  if (td->last_sys_num == static_cast<ADDRINT>(__NR_open)) {
    const char *pathPtr =
        reinterpret_cast<const char *>(td->last_sys_arg0);
    char tmp[512];
    tmp[0] = 0;
    PIN_SafeCopy(tmp, pathPtr, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;
    td->last_open_match = PathMatchesTaintFile(std::string(tmp));
  } else if (td->last_sys_num == static_cast<ADDRINT>(__NR_openat)) {
    const char *pathPtr =
        reinterpret_cast<const char *>(td->last_sys_arg1);
    char tmp[512];
    tmp[0] = 0;
    PIN_SafeCopy(tmp, pathPtr, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;
    td->last_open_match = PathMatchesTaintFile(std::string(tmp));
  }
}

static VOID SyscallExit(THREADID tid, CONTEXT *ctx, SYSCALL_STANDARD std,
                        VOID *) {
  if (!g_taint_enabled || g_seed_from_input_tensor) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  const ADDRINT num = td->last_sys_num;
  const ADDRINT ret = PIN_GetSyscallReturn(ctx, std);

  if (num == static_cast<ADDRINT>(__NR_open) ||
      num == static_cast<ADDRINT>(__NR_openat)) {
    if (td->last_open_match && static_cast<long>(ret) >= 0) {
      const int fd = static_cast<int>(ret);
      if (!g_taint_no_lock) {
        PIN_GetLock(&g_taint_lock, 1);
      }
      g_taint_fds.insert(fd);
      if (!g_taint_no_lock) {
        PIN_ReleaseLock(&g_taint_lock);
      }
    }
    return;
  }

  if (num == static_cast<ADDRINT>(__NR_close)) {
    const int fd = static_cast<int>(td->last_sys_arg0);
    if (!g_taint_no_lock) {
      PIN_GetLock(&g_taint_lock, 1);
    }
    g_taint_fds.erase(fd);
    if (!g_taint_no_lock) {
      PIN_ReleaseLock(&g_taint_lock);
    }
    return;
  }

  if (num == static_cast<ADDRINT>(__NR_mmap)) {
    const int fd = static_cast<int>(td->last_sys_arg4);
    const uint64_t length = static_cast<uint64_t>(td->last_sys_arg1);
    if (fd >= 0 && length > 0 && static_cast<long>(ret) >= 0 &&
        IsTaintFd(fd)) {
      MemSetTaintRange(static_cast<uint64_t>(ret), length, true);
    }
    return;
  }

  if (num == static_cast<ADDRINT>(__NR_munmap)) {
    const uint64_t addr = static_cast<uint64_t>(td->last_sys_arg0);
    const uint64_t length = static_cast<uint64_t>(td->last_sys_arg1);
    if (ret == 0 && length > 0) {
      MemSetTaintRange(addr, length, false);
    }
    return;
  }

  if (num == static_cast<ADDRINT>(__NR_read) ||
      num == static_cast<ADDRINT>(__NR_pread64)) {
    const int fd = static_cast<int>(td->last_sys_arg0);
    const uint64_t buf = static_cast<uint64_t>(td->last_sys_arg1);
    const long nread = static_cast<long>(ret);

    if (IsTaintFd(fd) && nread > 0 &&
        nread <= static_cast<long>(std::numeric_limits<uint32_t>::max())) {
      MemSetTaint(buf, static_cast<uint32_t>(nread), true);
    }
  }
}

static VOID saveWriteBefore(THREADID tid, UINT32 memOp, ADDRINT ea,
                            UINT32 size, ADDRINT ip) {
  if (size == 0 || memOp >= kMaxSavedMemOps) {
    return;
  }

  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  const uint64_t siteIp = static_cast<uint64_t>(ip);
  if (g_has_site_ip_filter && siteIp != g_site_ip_filter) {
    return;
  }
  if (g_has_caller_ip_filter && CurrentCallerIp(td) != g_caller_ip_filter) {
    return;
  }

  SavedWriteSlot &slot = td->saved[memOp];
  slot.valid = false;
  slot.writeAddr = static_cast<uint64_t>(ea);
  slot.writeSize = size;
  slot.blocks.clear();

  const uint64_t vaddr = static_cast<uint64_t>(ea);
  const uint64_t end = vaddr + static_cast<uint64_t>(size);
  const uint64_t start = vaddr & ~0xFULL;
  const uint64_t last = (end - 1) & ~0xFULL;
  const bool writeTainted = currentWriteIsTainted(td);

  for (uint64_t blockAddr = start; blockAddr <= last; blockAddr += kBlockSize) {
    if (!g_record_all_writes) {
      const bool blockWasTainted =
          MemAnyTaint(blockAddr, static_cast<uint32_t>(kBlockSize));
      if (!blockWasTainted && !writeTainted) {
        continue;
      }
    }

    SavedBlock block;
    block.blockAddr = blockAddr;
    const size_t copied =
        PIN_SafeCopy(block.before.data(),
                     reinterpret_cast<const VOID *>(blockAddr), kBlockSize);
    if (copied < kBlockSize) {
      std::memset(block.before.data() + copied, 0, kBlockSize - copied);
    }
    slot.blocks.push_back(block);
  }

  slot.valid = !slot.blocks.empty();
}

static VOID OnCall(THREADID tid, ADDRINT ip) {
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  td->callstack.push_back(static_cast<uint64_t>(ip));
}

static VOID OnRet(THREADID tid) {
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || td->callstack.empty()) {
    return;
  }
  td->callstack.pop_back();
}

static uint64_t CurrentCallerIp(const ThreadData *td) {
  if (td == nullptr || td->callstack.empty()) {
    return 0;
  }
  return td->callstack.back();
}

static VOID recordWriteAfter(THREADID tid, UINT32 memOp, ADDRINT ip) {
  if (memOp >= kMaxSavedMemOps) {
    return;
  }

  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  SavedWriteSlot &slot = td->saved[memOp];
  if (!slot.valid) {
    return;
  }

  std::map<uint64_t, SiteMeta>::const_iterator metaIt = g_siteMetaByIp.find(
      static_cast<uint64_t>(ip));
  SiteMeta meta = metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(ip)
                                                 : metaIt->second;
  const uint64_t callerIp = CurrentCallerIp(td);

  if ((g_has_site_ip_filter && meta.ip != g_site_ip_filter) ||
      (g_has_caller_ip_filter && callerIp != g_caller_ip_filter)) {
    slot.valid = false;
    slot.writeAddr = 0;
    slot.writeSize = 0;
    slot.blocks.clear();
    return;
  }

  PIN_GetLock(&g_stream_lock, tid + 1);
  for (size_t i = 0; i < slot.blocks.size(); ++i) {
    const SavedBlock &block = slot.blocks[i];
    const size_t copied =
        PIN_SafeCopy(td->after.data(),
                     reinterpret_cast<const VOID *>(block.blockAddr), kBlockSize);
    if (copied < kBlockSize) {
      std::memset(td->after.data() + copied, 0, kBlockSize - copied);
    }

    const bool changed =
        std::memcmp(block.before.data(), td->after.data(), kBlockSize) != 0;

    if (g_site_bits_only) {
      SiteState &siteState = g_sites[meta.ip];
      if (siteState.writeCount == 0) {
        siteState.ip = meta.ip;
      }
      siteState.writeCount += 1;
      if (changed) {
        siteState.changedCount += 1;
        siteState.bits.push_back('1');
      } else {
        siteState.unchangedCount += 1;
        siteState.bits.push_back('0');
      }
      g_siteBitsOnlyBlocks.insert(block.blockAddr);
      continue;
    }

    BlockState &state = g_blocks[block.blockAddr];
    if (state.writeCount == 0) {
      state.vaddr = block.blockAddr;
      state.hasInitialBaseline = true;
    }
    state.writeCount += 1;
    if (changed) {
      state.changedCount += 1;
      state.bits.push_back('1');
    } else {
      state.unchangedCount += 1;
      state.bits.push_back('0');
    }
    emitBlockEvent(slot, block, meta, state.writeCount, state.unchangedCount,
                   changed, td->after);
    if (!changed) {
      emitUnchangedEvent(slot, block, meta, state.writeCount,
                         state.unchangedCount, td->after);
    }
    state.ownerCounts[meta.ip] += 1;

    SiteState &siteState = g_sites[meta.ip];
    if (siteState.writeCount == 0) {
      siteState.ip = meta.ip;
    }
    siteState.writeCount += 1;
    if (changed) {
      siteState.changedCount += 1;
      siteState.bits.push_back('1');
    } else {
      siteState.unchangedCount += 1;
      siteState.bits.push_back('0');
    }
  }
  PIN_ReleaseLock(&g_stream_lock);

  slot.valid = false;
  slot.writeAddr = 0;
  slot.writeSize = 0;
  slot.blocks.clear();
}

static VOID Routine(RTN rtn, VOID *) {
  if (!RTN_Valid(rtn) || !g_seed_from_input_tensor) {
    return;
  }

  const std::string &name = RTN_Name(rtn);
  if (name.find("updateInputPlaceholders") != std::string::npos) {
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnEnterUpdateInputs),
                   IARG_THREAD_ID, IARG_END);
    RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(OnExitUpdateInputs),
                   IARG_THREAD_ID, IARG_END);
    RTN_Close(rtn);
    return;
  }

  if (name.find("Tensor6assign") != std::string::npos) {
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnEnterTensorAssign),
                   IARG_THREAD_ID, IARG_END);
    RTN_InsertCall(rtn, IPOINT_AFTER, AFUNPTR(OnExitTensorAssign),
                   IARG_THREAD_ID, IARG_END);
    RTN_Close(rtn);
  }
}

static VOID Instruction(INS ins, VOID *) {
  if (INS_IsCall(ins)) {
    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(OnCall), IARG_THREAD_ID,
                   IARG_ADDRINT, INS_Address(ins), IARG_END);
  } else if (INS_IsRet(ins)) {
    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(OnRet), IARG_THREAD_ID,
                   IARG_END);
  }

  if (INS_HasScatteredMemoryAccess(ins) || INS_IsVscatter(ins) ||
      INS_IsVgather(ins)) {
    return;
  }

  const UINT32 memOps = INS_MemoryOperandCount(ins);

  bool hasMemRead = false;
  bool hasMemWrite = false;
  for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
    if (INS_MemoryOperandIsRead(ins, memOp)) {
      hasMemRead = true;
    }
    if (INS_MemoryOperandIsWritten(ins, memOp)) {
      hasMemWrite = true;
    }
  }

  if (g_readEventBin.is_open() && hasMemRead) {
    const uint64_t insIp = static_cast<uint64_t>(INS_Address(ins));
    if (!g_has_read_site_ip_filters ||
        g_read_site_ip_filters.find(insIp) != g_read_site_ip_filters.end()) {
      for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
        if (!INS_MemoryOperandIsRead(ins, memOp)) {
          continue;
        }
        const UINT32 memOpSize =
            static_cast<UINT32>(INS_MemoryOperandSize(ins, memOp));
        if (memOpSize == 0) {
          continue;
        }
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(emitReadEvent),
                                 IARG_THREAD_ID, IARG_UINT32, memOp,
                                 IARG_MEMORYOP_EA, memOp, IARG_UINT32,
                                 memOpSize, IARG_INST_PTR, IARG_END);
      }
    }
  }

  if (g_taint_enabled) {
    const bool hasRegWrites = (INS_MaxNumWRegs(ins) > 0);
    if (hasRegWrites || hasMemWrite) {
      INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintBegin), IARG_THREAD_ID,
                     IARG_END);

      std::unordered_set<REG> addrRegs;
      const REG baseReg = NormReg(INS_MemoryBaseReg(ins));
      const REG indexReg = NormReg(INS_MemoryIndexReg(ins));
      if (baseReg != REG_INVALID()) {
        addrRegs.insert(baseReg);
      }
      if (indexReg != REG_INVALID()) {
        addrRegs.insert(indexReg);
      }

      const UINT32 maxReads = INS_MaxNumRRegs(ins);
      for (UINT32 i = 0; i < maxReads; ++i) {
        REG reg = NormReg(INS_RegR(ins, i));
        if (reg == REG_INVALID() || addrRegs.find(reg) != addrRegs.end()) {
          continue;
        }
        INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccReg),
                       IARG_THREAD_ID, IARG_UINT32,
                       static_cast<UINT32>(reg), IARG_END);
      }

      if (hasMemRead) {
        for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
          if (!INS_MemoryOperandIsRead(ins, memOp)) {
            continue;
          }
          const UINT32 memOpSize =
              static_cast<UINT32>(INS_MemoryOperandSize(ins, memOp));
          if (memOpSize == 0) {
            continue;
          }
          INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccMem),
                         IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp,
                         IARG_UINT32, memOpSize, IARG_END);
        }
      }

      const BOOL hasAfter = INS_IsValidForIpointAfter(ins);
      if (hasAfter) {
        for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
          if (!INS_MemoryOperandIsWritten(ins, memOp)) {
            continue;
          }
          const UINT32 memOpSize =
              static_cast<UINT32>(INS_MemoryOperandSize(ins, memOp));
          if (memOpSize == 0) {
            continue;
          }
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(saveWriteBefore),
                                   IARG_THREAD_ID, IARG_UINT32, memOp,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32,
                                   memOpSize, IARG_INST_PTR, IARG_END);
        }
      }

      const UINT32 maxWrites = INS_MaxNumWRegs(ins);
      for (UINT32 i = 0; i < maxWrites; ++i) {
        REG reg = NormReg(INS_RegW(ins, i));
        if (reg == REG_INVALID()) {
          continue;
        }
        INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetReg),
                       IARG_THREAD_ID, IARG_UINT32,
                       static_cast<UINT32>(reg), IARG_END);
      }

      for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
        if (!INS_MemoryOperandIsWritten(ins, memOp)) {
          continue;
        }
        const UINT32 memOpSize =
            static_cast<UINT32>(INS_MemoryOperandSize(ins, memOp));
        if (memOpSize == 0) {
          continue;
        }
        INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetMemSeedAware),
                       IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp,
                       IARG_UINT32, memOpSize, IARG_END);
      }

      if (hasAfter) {
        for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
          if (!INS_MemoryOperandIsWritten(ins, memOp)) {
            continue;
          }
          INS_InsertPredicatedCall(ins, IPOINT_AFTER, AFUNPTR(recordWriteAfter),
                                   IARG_THREAD_ID, IARG_UINT32, memOp,
                                   IARG_INST_PTR, IARG_END);
        }
      }
    }
  }

  if (!hasMemWrite) {
    return;
  }

  const SiteMeta meta = describeSite(ins);
  if (!g_site_bits_only || !g_has_site_ip_filter || meta.ip == g_site_ip_filter) {
    g_siteMetaByIp.emplace(meta.ip, meta);
  }

  if (!INS_IsValidForIpointAfter(ins)) {
    g_sitesWithoutAfter.insert(meta.site);
    return;
  }

  for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
    if (!INS_MemoryOperandIsWritten(ins, memOp)) {
      continue;
    }
    const UINT32 memOpSize =
        static_cast<UINT32>(INS_MemoryOperandSize(ins, memOp));
    if (memOpSize == 0) {
      continue;
    }
    INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(saveWriteBefore),
                             IARG_THREAD_ID, IARG_UINT32, memOp,
                             IARG_MEMORYOP_EA, memOp, IARG_UINT32, memOpSize,
                             IARG_INST_PTR, IARG_END);
    INS_InsertPredicatedCall(ins, IPOINT_AFTER, AFUNPTR(recordWriteAfter),
                             IARG_THREAD_ID, IARG_UINT32, memOp, IARG_INST_PTR,
                             IARG_END);
  }
}

static VOID ThreadStart(THREADID tid, CONTEXT *, INT32, VOID *) {
  ThreadData *td = new ThreadData();
  td->reg_taint.resize(static_cast<size_t>(REG_LAST));
  PIN_SetThreadData(g_tls_key, td, tid);
}

static VOID ThreadFini(THREADID tid, const CONTEXT *, INT32, VOID *) {
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  delete td;
  PIN_SetThreadData(g_tls_key, nullptr, tid);
}

static void writeJsonStringArray(std::ofstream &out,
                                 const std::vector<std::string> &values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << "\"" << jsonEscape(values[i]) << "\"";
  }
  out << "]";
}

static void writeOwnerObject(std::ofstream &out, const SiteMeta &meta,
                             uint64_t count) {
  out << "{\n";
  out << "          \"ip\": " << meta.ip << ",\n";
  out << "          \"ip_hex\": \"" << jsonEscape(meta.ipHex) << "\",\n";
  out << "          \"module\": \"" << jsonEscape(meta.image) << "\",\n";
  out << "          \"offset\": \"" << jsonEscape(meta.imageOffset) << "\",\n";
  out << "          \"symbol\": \"" << jsonEscape(meta.site) << "\",\n";
  out << "          \"routine\": \"" << jsonEscape(meta.routine) << "\",\n";
  out << "          \"disasm\": \"" << jsonEscape(meta.disasm) << "\",\n";
  out << "          \"write_count\": " << count << "\n";
  out << "        }";
}

static VOID Fini(INT32, VOID *) {
  std::ofstream out(KnobOutput.Value().c_str());
  if (!out) {
    std::fprintf(stderr, "failed to open output: %s\n",
                 KnobOutput.Value().c_str());
    return;
  }

  std::vector<std::string> sitesWithoutAfter(g_sitesWithoutAfter.begin(),
                                             g_sitesWithoutAfter.end());
  out << std::setprecision(17);

  if (g_site_bits_only) {
    out << "{\n";
    out << "  \"taint_file\": \"" << jsonEscape(g_taint_file) << "\",\n";
    out << "  \"taint_seed_mode\": \"" << jsonEscape(KnobTaintSeedMode.Value())
        << "\",\n";
    out << "  \"block_size\": " << kBlockSize << ",\n";
    out << "  \"read_event_bin\": \"" << jsonEscape(KnobReadEventBin.Value())
        << "\",\n";
    out << "  \"read_event_record_size\": " << sizeof(ReadEventRecord)
        << ",\n";
    out << "  \"read_event_count\": " << g_readEventCount << ",\n";
    out << "  \"site_bits_only\": true,\n";
    out << "  \"site_bits_txt\": \"" << jsonEscape(KnobSiteBitsText.Value())
        << "\",\n";
    out << "  \"all_writes\": " << (g_record_all_writes ? "true" : "false")
        << ",\n";
    out << "  \"address_count\": 0,\n";
    out << "  \"touched_block_count\": " << g_siteBitsOnlyBlocks.size()
        << ",\n";
    out << "  \"sites_without_after\": ";
    writeJsonStringArray(out, sitesWithoutAfter);
    out << ",\n";
    out << "  \"site_ip_filter\": "
        << (g_has_site_ip_filter ? std::to_string(g_site_ip_filter) : "null")
        << ",\n";
    out << "  \"site_ip_filter_hex\": "
        << (g_has_site_ip_filter ? ("\"" + hexAddr(g_site_ip_filter) + "\"")
                                 : "null")
        << ",\n";
    out << "  \"caller_ip_filter\": "
        << (g_has_caller_ip_filter ? std::to_string(g_caller_ip_filter) : "null")
        << ",\n";
    out << "  \"caller_ip_filter_hex\": "
        << (g_has_caller_ip_filter ? ("\"" + hexAddr(g_caller_ip_filter) + "\"")
                                   : "null")
        << ",\n";
    out << "  \"site_bit_definition\": "
        << "\"Per-exact-write-site 01 string over taint-filtered block-overlap "
           "writes. Each bit is attributed to the precise write IP that "
           "produced that before/after comparison for one overlapping 16-byte "
           "block. 0 means the block stayed identical, 1 means the block "
           "changed.\",\n";
    out << "  \"addresses\": [],\n";
    out << "  \"site_count\": " << g_sites.size() << ",\n";
    out << "  \"sites\": [\n";

    bool firstSite = true;
    std::string firstBits;
    for (std::map<uint64_t, SiteState>::const_iterator it = g_sites.begin();
         it != g_sites.end(); ++it) {
      const SiteState &state = it->second;
      const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
          g_siteMetaByIp.find(state.ip);
      const SiteMeta meta =
          metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(state.ip)
                                         : metaIt->second;
      const double unchangedRatio =
          state.writeCount == 0
              ? 0.0
              : static_cast<double>(state.unchangedCount) /
                    static_cast<double>(state.writeCount);

      if (firstSite) {
        firstBits = state.bits;
      } else {
        out << ",\n";
      }
      firstSite = false;

      out << "    {\n";
      out << "      \"ip\": " << meta.ip << ",\n";
      out << "      \"ip_hex\": \"" << jsonEscape(meta.ipHex) << "\",\n";
      out << "      \"module\": \"" << jsonEscape(meta.image) << "\",\n";
      out << "      \"offset\": \"" << jsonEscape(meta.imageOffset) << "\",\n";
      out << "      \"symbol\": \"" << jsonEscape(meta.site) << "\",\n";
      out << "      \"routine\": \"" << jsonEscape(meta.routine) << "\",\n";
      out << "      \"disasm\": \"" << jsonEscape(meta.disasm) << "\",\n";
      out << "      \"compare_count\": " << state.writeCount << ",\n";
      out << "      \"bits_len\": " << state.bits.size() << ",\n";
      out << "      \"unchanged_count\": " << state.unchangedCount << ",\n";
      out << "      \"changed_count\": " << state.changedCount << ",\n";
      out << "      \"unchanged_ratio\": " << unchangedRatio << ",\n";
      out << "      \"bits\": \"" << state.bits << "\"\n";
      out << "    }";
    }

    out << "\n  ]\n";
    out << "}\n";
    out.flush();

    if (!KnobSiteBitsText.Value().empty()) {
      std::ofstream bitsOut(KnobSiteBitsText.Value().c_str());
      if (!bitsOut) {
        std::fprintf(stderr, "failed to open site bits text output: %s\n",
                     KnobSiteBitsText.Value().c_str());
      } else if (!firstBits.empty()) {
        bitsOut << firstBits;
        bitsOut.flush();
      }
    }

    if (!KnobIpMap.Value().empty()) {
      std::ofstream ipMap(KnobIpMap.Value().c_str());
      if (ipMap) {
        ipMap << "# ip\timage\timage_offset\troutine\tdisasm\n";
        for (std::map<uint64_t, SiteMeta>::const_iterator it =
                 g_siteMetaByIp.begin();
             it != g_siteMetaByIp.end(); ++it) {
          const SiteMeta &meta = it->second;
          ipMap << meta.ipHex << "\t" << meta.image << "\t"
                << meta.imageOffset << "\t" << meta.site << "\t"
                << meta.disasm << "\n";
        }
        ipMap.flush();
      }
    }

    if (g_unchangedEventBin.is_open()) {
      g_unchangedEventBin.flush();
      g_unchangedEventBin.close();
    }
    if (g_blockEventBin.is_open()) {
      g_blockEventBin.flush();
      g_blockEventBin.close();
    }
    if (g_readEventBin.is_open()) {
      g_readEventBin.flush();
      g_readEventBin.close();
    }
    for (std::unordered_map<uint64_t, ShadowPage *>::iterator it =
             g_shadow_pages.begin();
         it != g_shadow_pages.end(); ++it) {
      delete it->second;
    }
    g_shadow_pages.clear();
    return;
  }

  out << "{\n";
  out << "  \"taint_file\": \"" << jsonEscape(g_taint_file) << "\",\n";
  out << "  \"taint_seed_mode\": \"" << jsonEscape(KnobTaintSeedMode.Value())
      << "\",\n";
  out << "  \"block_size\": " << kBlockSize << ",\n";
  out << "  \"block_event_bin\": \"" << jsonEscape(KnobBlockEventBin.Value())
      << "\",\n";
  out << "  \"block_event_record_size\": " << sizeof(BlockEventRecord)
      << ",\n";
  out << "  \"block_event_count\": " << g_blockEventCount << ",\n";
  out << "  \"unchanged_event_bin\": \""
      << jsonEscape(KnobUnchangedEventBin.Value()) << "\",\n";
  out << "  \"unchanged_event_record_size\": "
      << sizeof(UnchangedEventRecord) << ",\n";
  out << "  \"unchanged_event_count\": " << g_unchangedEventCount << ",\n";
  out << "  \"read_event_bin\": \"" << jsonEscape(KnobReadEventBin.Value())
      << "\",\n";
  out << "  \"read_event_record_size\": " << sizeof(ReadEventRecord) << ",\n";
  out << "  \"read_event_count\": " << g_readEventCount << ",\n";
  out << "  \"address_count\": " << g_blocks.size() << ",\n";
  out << "  \"sites_without_after\": ";
  writeJsonStringArray(out, sitesWithoutAfter);
  out << ",\n";
  out << "  \"bit_definition\": "
      << "\"Per-tainted-16B-address 01 string over writes. Each bit compares "
         "the full 16-byte aligned block immediately before and immediately "
         "after one write that overlaps that block. 0 means the block stayed "
         "identical, 1 means the block changed. The first bit therefore "
         "compares the block's initial pre-write value against the value after "
         "the first recorded write.\",\n";
  out << "  \"site_bit_definition\": "
      << "\"Per-exact-write-site 01 string over taint-filtered block-overlap "
         "writes. Each bit is attributed to the precise write IP that produced "
         "that before/after comparison for one overlapping 16-byte block. 0 "
         "means the block stayed identical, 1 means the block changed.\",\n";
  out << "  \"addresses\": [\n";

  bool firstBlock = true;
  for (std::map<uint64_t, BlockState>::const_iterator it = g_blocks.begin();
       it != g_blocks.end(); ++it) {
    const BlockState &state = it->second;

    std::vector<std::pair<uint64_t, uint64_t> > owners(state.ownerCounts.begin(),
                                                       state.ownerCounts.end());
    std::sort(owners.begin(), owners.end(),
              [](const std::pair<uint64_t, uint64_t> &a,
                 const std::pair<uint64_t, uint64_t> &b) {
                if (a.second != b.second) {
                  return a.second > b.second;
                }
                return a.first < b.first;
              });

    const double unchangedRatio =
        state.writeCount == 0
            ? 0.0
            : static_cast<double>(state.unchangedCount) /
                  static_cast<double>(state.writeCount);

    if (!firstBlock) {
      out << ",\n";
    }
    firstBlock = false;

    out << "    {\n";
    out << "      \"vaddr\": " << state.vaddr << ",\n";
    out << "      \"vaddr_hex\": \"" << hexAddr(state.vaddr) << "\",\n";
    out << "      \"size\": " << kBlockSize << ",\n";
    out << "      \"write_count\": " << state.writeCount << ",\n";
    out << "      \"compare_count\": " << state.writeCount << ",\n";
    out << "      \"bits_len\": " << state.bits.size() << ",\n";
    out << "      \"has_initial_baseline\": "
        << (state.hasInitialBaseline ? "true" : "false") << ",\n";
    out << "      \"unchanged_count\": " << state.unchangedCount << ",\n";
    out << "      \"changed_count\": " << state.changedCount << ",\n";
    out << "      \"unchanged_ratio\": " << unchangedRatio << ",\n";
    out << "      \"bits\": \"" << state.bits << "\",\n";

    if (!owners.empty()) {
      const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
          g_siteMetaByIp.find(owners.front().first);
      const SiteMeta topMeta =
          metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(owners.front().first)
                                         : metaIt->second;
      out << "      \"owner\": {\n";
      out << "        \"ip\": " << topMeta.ip << ",\n";
      out << "        \"ip_hex\": \"" << jsonEscape(topMeta.ipHex) << "\",\n";
      out << "        \"module\": \"" << jsonEscape(topMeta.image) << "\",\n";
      out << "        \"offset\": \"" << jsonEscape(topMeta.imageOffset)
          << "\",\n";
      out << "        \"symbol\": \"" << jsonEscape(topMeta.site) << "\",\n";
      out << "        \"routine\": \"" << jsonEscape(topMeta.routine) << "\",\n";
      out << "        \"disasm\": \"" << jsonEscape(topMeta.disasm) << "\",\n";
      out << "        \"write_count\": " << owners.front().second << "\n";
      out << "      },\n";
    } else {
      out << "      \"owner\": null,\n";
    }

    out << "      \"owner_candidate_count\": " << owners.size() << ",\n";
    out << "      \"owner_candidates\": [\n";
    for (size_t i = 0; i < owners.size(); ++i) {
      const uint64_t ip = owners[i].first;
      const uint64_t count = owners[i].second;
      const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
          g_siteMetaByIp.find(ip);
      const SiteMeta ownerMeta =
          metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(ip) : metaIt->second;
      if (i != 0) {
        out << ",\n";
      }
      writeOwnerObject(out, ownerMeta, count);
    }
    out << "\n";
    out << "      ]\n";
    out << "    }";
  }

  out << "\n  ],\n";
  out << "  \"site_count\": " << g_sites.size() << ",\n";
  out << "  \"all_writes\": " << (g_record_all_writes ? "true" : "false")
      << ",\n";
  out << "  \"site_ip_filter\": "
      << (g_has_site_ip_filter ? std::to_string(g_site_ip_filter) : "null")
      << ",\n";
  out << "  \"site_ip_filter_hex\": "
      << (g_has_site_ip_filter ? ("\"" + hexAddr(g_site_ip_filter) + "\"")
                               : "null")
      << ",\n";
  out << "  \"caller_ip_filter\": "
      << (g_has_caller_ip_filter ? std::to_string(g_caller_ip_filter) : "null")
      << ",\n";
  out << "  \"caller_ip_filter_hex\": "
      << (g_has_caller_ip_filter ? ("\"" + hexAddr(g_caller_ip_filter) + "\"")
                                 : "null")
      << ",\n";
  out << "  \"sites\": [\n";

  bool firstSite = true;
  for (std::map<uint64_t, SiteState>::const_iterator it = g_sites.begin();
       it != g_sites.end(); ++it) {
    const SiteState &state = it->second;
    const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
        g_siteMetaByIp.find(state.ip);
    const SiteMeta meta =
        metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(state.ip) : metaIt->second;
    const double unchangedRatio =
        state.writeCount == 0
            ? 0.0
            : static_cast<double>(state.unchangedCount) /
                  static_cast<double>(state.writeCount);

    if (!firstSite) {
      out << ",\n";
    }
    firstSite = false;

    out << "    {\n";
    out << "      \"ip\": " << meta.ip << ",\n";
    out << "      \"ip_hex\": \"" << jsonEscape(meta.ipHex) << "\",\n";
    out << "      \"module\": \"" << jsonEscape(meta.image) << "\",\n";
    out << "      \"offset\": \"" << jsonEscape(meta.imageOffset) << "\",\n";
    out << "      \"symbol\": \"" << jsonEscape(meta.site) << "\",\n";
    out << "      \"routine\": \"" << jsonEscape(meta.routine) << "\",\n";
    out << "      \"disasm\": \"" << jsonEscape(meta.disasm) << "\",\n";
    out << "      \"compare_count\": " << state.writeCount << ",\n";
    out << "      \"bits_len\": " << state.bits.size() << ",\n";
    out << "      \"unchanged_count\": " << state.unchangedCount << ",\n";
    out << "      \"changed_count\": " << state.changedCount << ",\n";
    out << "      \"unchanged_ratio\": " << unchangedRatio << ",\n";
    out << "      \"bits\": \"" << state.bits << "\"\n";
    out << "    }";
  }

  out << "\n  ]\n";
  out << "}\n";
  out.flush();

  if (!KnobIpMap.Value().empty()) {
    std::ofstream ipMap(KnobIpMap.Value().c_str());
    if (ipMap) {
      ipMap << "# ip\timage\timage_offset\troutine\tdisasm\n";
      for (std::map<uint64_t, SiteMeta>::const_iterator it = g_siteMetaByIp.begin();
           it != g_siteMetaByIp.end(); ++it) {
        const SiteMeta &meta = it->second;
        ipMap << meta.ipHex << "\t" << meta.image << "\t" << meta.imageOffset
              << "\t" << meta.site << "\t" << meta.disasm << "\n";
      }
      ipMap.flush();
    }
  }

  if (g_unchangedEventBin.is_open()) {
    g_unchangedEventBin.flush();
    g_unchangedEventBin.close();
  }
  if (g_blockEventBin.is_open()) {
    g_blockEventBin.flush();
    g_blockEventBin.close();
  }
  if (g_readEventBin.is_open()) {
    g_readEventBin.flush();
    g_readEventBin.close();
  }

  for (std::unordered_map<uint64_t, ShadowPage *>::iterator it =
           g_shadow_pages.begin();
       it != g_shadow_pages.end(); ++it) {
    delete it->second;
  }
  g_shadow_pages.clear();
}

} // namespace

int main(int argc, char *argv[]) {
  PIN_InitSymbols();
  if (PIN_Init(argc, argv)) {
    return Usage();
  }

  g_taint_file = KnobTaintFile.Value();
  g_taint_no_lock = (KnobTaintNoLock.Value() != 0);
  g_seed_from_input_tensor = (KnobTaintSeedMode.Value() == "input-tensor");
  g_taint_enabled = (!g_taint_file.empty()) || g_seed_from_input_tensor;
  if (!g_taint_file.empty()) {
    g_taint_file_base = BaseName(g_taint_file);
  }
  g_record_all_writes = (KnobAllWrites.Value() != 0);
  g_site_bits_only = (KnobSiteBitsOnly.Value() != 0);
  if (!KnobSiteIp.Value().empty()) {
    if (!parseUint64Arg(KnobSiteIp.Value(), &g_site_ip_filter)) {
      std::fprintf(stderr, "invalid -site-ip: %s\n", KnobSiteIp.Value().c_str());
      return 1;
    }
    g_has_site_ip_filter = true;
  }
  if (!KnobCallerIp.Value().empty()) {
    if (!parseUint64Arg(KnobCallerIp.Value(), &g_caller_ip_filter)) {
      std::fprintf(stderr, "invalid -caller-ip: %s\n",
                   KnobCallerIp.Value().c_str());
      return 1;
    }
    g_has_caller_ip_filter = true;
  }
  if (!KnobReadSiteIpList.Value().empty()) {
    if (!parseUint64ListArg(KnobReadSiteIpList.Value(),
                            &g_read_site_ip_filters)) {
      std::fprintf(stderr, "invalid -read-site-ip-list: %s\n",
                   KnobReadSiteIpList.Value().c_str());
      return 1;
    }
    g_has_read_site_ip_filters = true;
  }
  if (!KnobReadEventBin.Value().empty() && !g_has_read_site_ip_filters) {
    std::fprintf(stderr,
                 "-read-event-bin requires -read-site-ip-list to bound output\n");
    return 1;
  }
  if (g_site_bits_only && !g_has_site_ip_filter) {
    std::fprintf(stderr, "-site-bits-only requires -site-ip\n");
    return 1;
  }
  if (g_site_bits_only &&
      (!KnobBlockEventBin.Value().empty() || !KnobUnchangedEventBin.Value().empty())) {
    std::fprintf(stderr,
                 "-site-bits-only does not support -block-event-bin or "
                 "-unchanged-event-bin\n");
    return 1;
  }

  if (!KnobReadEventBin.Value().empty()) {
    g_readEventBin.open(KnobReadEventBin.Value().c_str(),
                        std::ios::out | std::ios::binary | std::ios::trunc);
    if (!g_readEventBin) {
      std::fprintf(stderr, "failed to open read event output: %s\n",
                   KnobReadEventBin.Value().c_str());
      return 1;
    }

    ReadEventFileHeader header{};
    std::memcpy(header.magic, "TB16REV", 7);
    header.version = 1;
    header.recordSize = sizeof(ReadEventRecord);
    g_readEventBin.write(reinterpret_cast<const char *>(&header),
                         sizeof(header));
    if (!g_readEventBin) {
      std::fprintf(stderr, "failed to write read event header: %s\n",
                   KnobReadEventBin.Value().c_str());
      return 1;
    }
  }

  if (!g_site_bits_only && !KnobBlockEventBin.Value().empty()) {
    g_blockEventBin.open(KnobBlockEventBin.Value().c_str(),
                         std::ios::out | std::ios::binary | std::ios::trunc);
    if (!g_blockEventBin) {
      std::fprintf(stderr, "failed to open block event output: %s\n",
                   KnobBlockEventBin.Value().c_str());
      return 1;
    }

    BlockEventFileHeader header{};
    std::memcpy(header.magic, "TB16BEV", 7);
    header.version = 1;
    header.recordSize = sizeof(BlockEventRecord);
    header.blockSize = kBlockSize;
    g_blockEventBin.write(reinterpret_cast<const char *>(&header),
                          sizeof(header));
    if (!g_blockEventBin) {
      std::fprintf(stderr, "failed to write block event header: %s\n",
                   KnobBlockEventBin.Value().c_str());
      return 1;
    }
  }

  if (!g_site_bits_only && !KnobUnchangedEventBin.Value().empty()) {
    g_unchangedEventBin.open(KnobUnchangedEventBin.Value().c_str(),
                             std::ios::out | std::ios::binary | std::ios::trunc);
    if (!g_unchangedEventBin) {
      std::fprintf(stderr, "failed to open unchanged event output: %s\n",
                   KnobUnchangedEventBin.Value().c_str());
      return 1;
    }

    UnchangedEventFileHeader header{};
    std::memcpy(header.magic, "TB16UEV", 7);
    header.version = 1;
    header.recordSize = sizeof(UnchangedEventRecord);
    header.blockSize = kBlockSize;
    g_unchangedEventBin.write(reinterpret_cast<const char *>(&header),
                              sizeof(header));
    if (!g_unchangedEventBin) {
      std::fprintf(stderr, "failed to write unchanged event header: %s\n",
                   KnobUnchangedEventBin.Value().c_str());
      return 1;
    }
  }

  PIN_InitLock(&g_stream_lock);
  PIN_InitLock(&g_taint_lock);
  g_tls_key = PIN_CreateThreadDataKey(nullptr);

  RTN_AddInstrumentFunction(Routine, nullptr);
  INS_AddInstrumentFunction(Instruction, nullptr);
  PIN_AddSyscallEntryFunction(SyscallEntry, nullptr);
  PIN_AddSyscallExitFunction(SyscallExit, nullptr);
  PIN_AddThreadStartFunction(ThreadStart, nullptr);
  PIN_AddThreadFiniFunction(ThreadFini, nullptr);
  PIN_AddFiniFunction(Fini, nullptr);

  PIN_StartProgram();
  return 0;
}
