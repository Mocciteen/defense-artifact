#include "pin.H"

#include <unistd.h>
#include <sys/syscall.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    KNOB_MODE_WRITEONCE, "pintool", "o", "single_block_trace.json",
    "Output JSON for a single tainted 16B block");
static KNOB<std::string> KnobIpMap(
    KNOB_MODE_WRITEONCE, "pintool", "m", "",
    "Optional IP map output. Written as text for easier disassembly lookup.");
static KNOB<std::string> KnobBlockAddr(
    KNOB_MODE_WRITEONCE, "pintool", "block-addr", "",
    "Target 16B block virtual address. Parsed with base 0 and aligned down to "
    "16 bytes.");
static KNOB<std::string> KnobSiteAllowlistFile(
    KNOB_MODE_WRITEONCE, "pintool", "site-allowlist-file", "",
    "Optional file with one write-site label per line. If provided and "
    "-block-addr is empty, the tool tracks the dominant 16B block touched by "
    "the allowlisted sites.");
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

static PIN_LOCK g_stream_lock;
static PIN_LOCK g_taint_lock;
static TLS_KEY g_tls_key = INVALID_TLS_KEY;

static bool g_taint_enabled = false;
static bool g_taint_no_lock = false;
static bool g_seed_from_input_tensor = false;
static std::string g_taint_file;
static std::string g_taint_file_base;
static std::unordered_set<int> g_taint_fds;

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
  std::vector<SavedBlock> blocks;
};

struct ThreadData {
  std::array<SavedWriteSlot, kMaxSavedMemOps> saved{};
  std::array<uint8_t, kBlockSize> after{};

  bool cur_taint = false;
  std::vector<uint8_t> reg_taint;

  uint32_t update_inputs_depth = 0;
  uint32_t tensor_assign_depth = 0;

  ADDRINT last_sys_num = 0;
  ADDRINT last_sys_arg0 = 0;
  ADDRINT last_sys_arg1 = 0;
  ADDRINT last_sys_arg2 = 0;
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

static std::unordered_map<uint64_t, ShadowPage *> g_shadow_pages;
static std::map<uint64_t, SiteMeta> g_siteMetaByIp;
static std::map<uint64_t, BlockState> g_blocks;
static uint64_t g_targetBlockAddr = 0;
static bool g_selectByBlockAddr = false;
static bool g_selectBySiteAllowlist = false;
static std::set<uint64_t> g_candidateBlocks;
static std::string g_siteAllowlistFile;
static std::unordered_set<std::string> g_siteAllowlist;
static std::unordered_set<uint64_t> g_trackedSiteIps;

static size_t countTrackedOwners(const BlockState &state) {
  size_t hits = 0;
  for (std::map<uint64_t, uint64_t>::const_iterator it =
           state.ownerCounts.begin();
       it != state.ownerCounts.end(); ++it) {
    if (g_trackedSiteIps.find(it->first) != g_trackedSiteIps.end()) {
      ++hits;
    }
  }
  return hits;
}

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

static bool parseBlockAddr(const std::string &text, uint64_t *out) {
  if (out == nullptr) {
    return false;
  }
  if (text.empty()) {
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

static std::string trim(const std::string &text) {
  size_t begin = 0;
  while (begin < text.size() &&
         (text[begin] == ' ' || text[begin] == '\t' || text[begin] == '\r' ||
          text[begin] == '\n')) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin &&
         (text[end - 1] == ' ' || text[end - 1] == '\t' ||
          text[end - 1] == '\r' || text[end - 1] == '\n')) {
    --end;
  }
  return text.substr(begin, end - begin);
}

static bool loadSiteAllowlist(const std::string &path) {
  std::ifstream handle(path.c_str());
  if (!handle) {
    return false;
  }
  std::string line;
  while (std::getline(handle, line)) {
    const size_t commentPos = line.find('#');
    if (commentPos != std::string::npos) {
      line.erase(commentPos);
    }
    const std::string item = trim(line);
    if (!item.empty()) {
      g_siteAllowlist.insert(item);
    }
  }
  return true;
}

static INT32 Usage() {
  std::fprintf(
      stderr,
      "singleblock16trace: emit a 01 stream for one tainted 16B block\n"
      "  -block-addr <addr>              target 16B block address\n"
      "  -site-allowlist-file <path>     select the dominant block touched by "
      "allowlisted sites\n"
      "  -o <file>                       output JSON "
      "(default: single_block_trace.json)\n"
      "  -m <file>                       optional IP map output\n"
      "  -taint-file <path>              seed taint from read() of this file\n"
      "  -taint-seed-mode file|input-tensor  taint seed mode\n"
      "  -taint-no-lock 0|1              disable taint shadow locks\n");
  return -1;
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
  td->last_open_match = false;

  if (td->last_sys_num == static_cast<ADDRINT>(__NR_open)) {
    const char *pathPtr = reinterpret_cast<const char *>(td->last_sys_arg0);
    char tmp[512];
    tmp[0] = 0;
    PIN_SafeCopy(tmp, pathPtr, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;
    td->last_open_match = PathMatchesTaintFile(std::string(tmp));
  } else if (td->last_sys_num == static_cast<ADDRINT>(__NR_openat)) {
    const char *pathPtr = reinterpret_cast<const char *>(td->last_sys_arg1);
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

  if (num == static_cast<ADDRINT>(__NR_read) ||
      num == static_cast<ADDRINT>(__NR_pread64)) {
    const int fd = static_cast<int>(td->last_sys_arg0);
    const uint64_t buf = static_cast<uint64_t>(td->last_sys_arg1);
    const long nread = static_cast<long>(ret);

    bool isTaintFd = false;
    if (!g_taint_no_lock) {
      PIN_GetLock(&g_taint_lock, 1);
    }
    isTaintFd = (g_taint_fds.find(fd) != g_taint_fds.end());
    if (!g_taint_no_lock) {
      PIN_ReleaseLock(&g_taint_lock);
    }

    if (isTaintFd && nread > 0 &&
        nread <= static_cast<long>(std::numeric_limits<uint32_t>::max())) {
      MemSetTaint(buf, static_cast<uint32_t>(nread), true);
    }
  }
}

static VOID saveWriteBefore(THREADID tid, UINT32 memOp, ADDRINT ea,
                            UINT32 size, ADDRINT ip) {
  if ((!g_selectByBlockAddr && !g_selectBySiteAllowlist) || size == 0 ||
      memOp >= kMaxSavedMemOps) {
    return;
  }

  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  SavedWriteSlot &slot = td->saved[memOp];
  slot.valid = false;
  slot.blocks.clear();

  const uint64_t vaddr = static_cast<uint64_t>(ea);
  const uint64_t end = vaddr + static_cast<uint64_t>(size);
  const uint64_t start = vaddr & ~0xFULL;
  const uint64_t last = (end - 1) & ~0xFULL;
  const bool writeTainted = currentWriteIsTainted(td);

  if (g_selectBySiteAllowlist &&
      g_trackedSiteIps.find(static_cast<uint64_t>(ip)) != g_trackedSiteIps.end()) {
    PIN_GetLock(&g_stream_lock, tid + 1);
    for (uint64_t blockAddr = start; blockAddr <= last; blockAddr += kBlockSize) {
      g_candidateBlocks.insert(blockAddr);
    }
    PIN_ReleaseLock(&g_stream_lock);
  }

  for (uint64_t blockAddr = start; blockAddr <= last; blockAddr += kBlockSize) {
    if (g_selectByBlockAddr && blockAddr != g_targetBlockAddr) {
      continue;
    }

    if (g_selectBySiteAllowlist) {
      bool foundCandidate = false;
      PIN_GetLock(&g_stream_lock, tid + 1);
      for (std::set<uint64_t>::const_iterator it = g_candidateBlocks.begin();
           it != g_candidateBlocks.end(); ++it) {
        if (*it == blockAddr) {
          foundCandidate = true;
          break;
        }
      }
      PIN_ReleaseLock(&g_stream_lock);
      if (!foundCandidate) {
        continue;
      }
    }

    const bool blockWasTainted =
        MemAnyTaint(blockAddr, static_cast<uint32_t>(kBlockSize));
    if (!blockWasTainted && !writeTainted) {
      continue;
    }

    SavedBlock block;
    block.blockAddr = blockAddr;
    const size_t copied = PIN_SafeCopy(block.before.data(),
                                       reinterpret_cast<const VOID *>(blockAddr),
                                       kBlockSize);
    if (copied < kBlockSize) {
      std::memset(block.before.data() + copied, 0, kBlockSize - copied);
    }
    slot.blocks.push_back(block);
  }

  slot.valid = !slot.blocks.empty();
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

  const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
      g_siteMetaByIp.find(static_cast<uint64_t>(ip));
  const SiteMeta meta =
      metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(ip) : metaIt->second;

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
    state.ownerCounts[meta.ip] += 1;
  }
  PIN_ReleaseLock(&g_stream_lock);

  slot.valid = false;
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

  if (hasMemWrite) {
    const SiteMeta meta = describeSite(ins);
    g_siteMetaByIp.emplace(meta.ip, meta);
    if (g_selectBySiteAllowlist &&
        g_siteAllowlist.find(meta.site) != g_siteAllowlist.end()) {
      g_trackedSiteIps.insert(meta.ip);
    }
  }

  const bool hasRegWrites = (INS_MaxNumWRegs(ins) > 0);
  if (!g_taint_enabled || !(hasRegWrites || hasMemWrite)) {
    return;
  }

  const BOOL hasAfter = INS_IsValidForIpointAfter(ins);
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
    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccReg), IARG_THREAD_ID,
                   IARG_UINT32, static_cast<UINT32>(reg), IARG_END);
  }

  if (hasMemRead) {
    for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
      if (!INS_MemoryOperandIsRead(ins, memOp)) {
        continue;
      }
      INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccMem), IARG_THREAD_ID,
                     IARG_MEMORYOP_EA, memOp, IARG_MEMORYOP_SIZE, memOp,
                     IARG_END);
    }
  }

  if (hasAfter && hasMemWrite) {
    for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
      if (!INS_MemoryOperandIsWritten(ins, memOp)) {
        continue;
      }
      INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(saveWriteBefore),
                               IARG_THREAD_ID, IARG_UINT32, memOp,
                               IARG_MEMORYOP_EA, memOp, IARG_MEMORYOP_SIZE,
                               memOp, IARG_INST_PTR, IARG_END);
    }
  }

  const UINT32 maxWrites = INS_MaxNumWRegs(ins);
  for (UINT32 i = 0; i < maxWrites; ++i) {
    REG reg = NormReg(INS_RegW(ins, i));
    if (reg == REG_INVALID()) {
      continue;
    }
    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetReg), IARG_THREAD_ID,
                   IARG_UINT32, static_cast<UINT32>(reg), IARG_END);
  }

  if (hasMemWrite) {
    for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
      if (!INS_MemoryOperandIsWritten(ins, memOp)) {
        continue;
      }
      INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetMemSeedAware),
                     IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp,
                     IARG_MEMORYOP_SIZE, memOp, IARG_END);
    }
  }

  if (hasAfter && hasMemWrite) {
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

static void writeOwnerObject(std::ofstream &out, const SiteMeta &meta,
                             uint64_t count, const char *indent) {
  out << indent << "{\n";
  out << indent << "  \"ip\": " << meta.ip << ",\n";
  out << indent << "  \"ip_hex\": \"" << jsonEscape(meta.ipHex) << "\",\n";
  out << indent << "  \"module\": \"" << jsonEscape(meta.image) << "\",\n";
  out << indent << "  \"offset\": \"" << jsonEscape(meta.imageOffset)
      << "\",\n";
  out << indent << "  \"symbol\": \"" << jsonEscape(meta.site) << "\",\n";
  out << indent << "  \"routine\": \"" << jsonEscape(meta.routine) << "\",\n";
  out << indent << "  \"disasm\": \"" << jsonEscape(meta.disasm) << "\",\n";
  out << indent << "  \"write_count\": " << count << "\n";
  out << indent << "}";
}

static VOID Fini(INT32, VOID *) {
  std::ofstream out(KnobOutput.Value().c_str());
  if (!out) {
    std::fprintf(stderr, "failed to open output: %s\n",
                 KnobOutput.Value().c_str());
    return;
  }

  out << std::setprecision(17);
  out << "{\n";
  out << "  \"taint_file\": \"" << jsonEscape(g_taint_file) << "\",\n";
  out << "  \"taint_seed_mode\": \"" << jsonEscape(KnobTaintSeedMode.Value())
      << "\",\n";
  out << "  \"block_size\": " << kBlockSize << ",\n";
  out << "  \"selection_mode\": \"";
  if (g_selectByBlockAddr) {
    out << "block-addr";
  } else {
    out << "site-allowlist-top-block";
  }
  out << "\",\n";
  out << "  \"site_allowlist_file\": \"" << jsonEscape(g_siteAllowlistFile)
      << "\",\n";
  out << "  \"target_vaddr\": " << g_targetBlockAddr << ",\n";
  out << "  \"target_vaddr_hex\": \"" << jsonEscape(hexAddr(g_targetBlockAddr))
      << "\",\n";
  out << "  \"candidate_block_count_seen\": " << g_candidateBlocks.size()
      << ",\n";
  out << "  \"candidate_block_count\": " << g_blocks.size() << ",\n";
  out << "  \"bit_definition\": ";
  if (g_selectByBlockAddr) {
    out << "\"Single-block 01 string over writes. Each bit compares the full "
           "16-byte aligned target block immediately before and immediately "
           "after one overlapping tainted write. 0 means the block stayed "
           "identical, 1 means the block changed.\",\n";
  } else {
    out << "\"Single-block 01 string over writes to candidate blocks "
           "discovered from allowlisted sites. Each bit compares the full "
           "16-byte aligned selected block immediately before and immediately "
           "after one overlapping tainted write. 0 means the block stayed "
           "identical, 1 means the block changed.\",\n";
  }

  const BlockState *selected = nullptr;
  if (g_selectByBlockAddr) {
    const std::map<uint64_t, BlockState>::const_iterator it =
        g_blocks.find(g_targetBlockAddr);
    if (it != g_blocks.end()) {
      selected = &it->second;
    }
  } else if (!g_blocks.empty()) {
    for (std::map<uint64_t, BlockState>::const_iterator it = g_blocks.begin();
         it != g_blocks.end(); ++it) {
      if (selected == nullptr) {
        selected = &it->second;
        continue;
      }
      if (g_selectBySiteAllowlist) {
        const size_t candTrackedOwners = countTrackedOwners(it->second);
        const size_t bestTrackedOwners = countTrackedOwners(*selected);
        if (candTrackedOwners > bestTrackedOwners ||
            (candTrackedOwners == bestTrackedOwners &&
             it->second.writeCount > selected->writeCount) ||
            (candTrackedOwners == bestTrackedOwners &&
             it->second.writeCount == selected->writeCount &&
             it->second.vaddr < selected->vaddr)) {
          selected = &it->second;
        }
      } else if (it->second.writeCount > selected->writeCount ||
                 (it->second.writeCount == selected->writeCount &&
                  it->second.vaddr < selected->vaddr)) {
        selected = &it->second;
      }
    }
  }

  if (selected == nullptr) {
    out << "  \"block_found\": false,\n";
    out << "  \"block\": null\n";
  } else {
    std::vector<std::pair<uint64_t, uint64_t> > owners(selected->ownerCounts.begin(),
                                                       selected->ownerCounts.end());
    std::sort(owners.begin(), owners.end(),
              [](const std::pair<uint64_t, uint64_t> &a,
                 const std::pair<uint64_t, uint64_t> &b) {
                if (a.second != b.second) {
                  return a.second > b.second;
                }
                return a.first < b.first;
              });

    const double unchangedRatio =
        static_cast<double>(selected->unchangedCount) /
        static_cast<double>(selected->writeCount);

    out << "  \"block_found\": true,\n";
    out << "  \"block\": {\n";
    out << "    \"vaddr\": " << selected->vaddr << ",\n";
    out << "    \"vaddr_hex\": \"" << hexAddr(selected->vaddr) << "\",\n";
    out << "    \"size\": " << kBlockSize << ",\n";
    out << "    \"write_count\": " << selected->writeCount << ",\n";
    out << "    \"compare_count\": " << selected->writeCount << ",\n";
    out << "    \"bits_len\": " << selected->bits.size() << ",\n";
    out << "    \"has_initial_baseline\": "
        << (selected->hasInitialBaseline ? "true" : "false") << ",\n";
    out << "    \"unchanged_count\": " << selected->unchangedCount << ",\n";
    out << "    \"changed_count\": " << selected->changedCount << ",\n";
    out << "    \"unchanged_ratio\": " << unchangedRatio << ",\n";
    out << "    \"bits\": \"" << selected->bits << "\",\n";

    if (!owners.empty()) {
      const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
          g_siteMetaByIp.find(owners.front().first);
      const SiteMeta topMeta =
          metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(owners.front().first)
                                         : metaIt->second;
      out << "    \"owner\": {\n";
      out << "      \"ip\": " << topMeta.ip << ",\n";
      out << "      \"ip_hex\": \"" << jsonEscape(topMeta.ipHex) << "\",\n";
      out << "      \"module\": \"" << jsonEscape(topMeta.image) << "\",\n";
      out << "      \"offset\": \"" << jsonEscape(topMeta.imageOffset)
          << "\",\n";
      out << "      \"symbol\": \"" << jsonEscape(topMeta.site) << "\",\n";
      out << "      \"routine\": \"" << jsonEscape(topMeta.routine)
          << "\",\n";
      out << "      \"disasm\": \"" << jsonEscape(topMeta.disasm) << "\",\n";
      out << "      \"write_count\": " << owners.front().second << "\n";
      out << "    },\n";
    } else {
      out << "    \"owner\": null,\n";
    }

    out << "    \"owner_candidate_count\": " << owners.size() << ",\n";
    out << "    \"owner_candidates\": [\n";
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
      writeOwnerObject(out, ownerMeta, count, "      ");
    }
    out << "\n";
    out << "    ]\n";
    out << "  }\n";
  }
  out << "}\n";
  out.flush();

  if (!KnobIpMap.Value().empty() && selected != nullptr &&
      !selected->ownerCounts.empty()) {
    std::ofstream ipMap(KnobIpMap.Value().c_str());
    if (ipMap) {
      ipMap << "# ip\timage\timage_offset\troutine\tdisasm\n";
      for (std::map<uint64_t, uint64_t>::const_iterator it =
               selected->ownerCounts.begin();
           it != selected->ownerCounts.end(); ++it) {
        const uint64_t ip = it->first;
        const std::map<uint64_t, SiteMeta>::const_iterator metaIt =
            g_siteMetaByIp.find(ip);
        const SiteMeta meta =
            metaIt == g_siteMetaByIp.end() ? defaultSiteMeta(ip) : metaIt->second;
        ipMap << meta.ipHex << "\t" << meta.image << "\t" << meta.imageOffset
              << "\t" << meta.site << "\t" << meta.disasm << "\n";
      }
      ipMap.flush();
    }
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

  const std::string blockAddrText = KnobBlockAddr.Value();
  if (!blockAddrText.empty()) {
    uint64_t parsedBlockAddr = 0;
    if (!parseBlockAddr(blockAddrText, &parsedBlockAddr)) {
      std::fprintf(stderr, "invalid -block-addr: %s\n", blockAddrText.c_str());
      return Usage();
    }
    g_targetBlockAddr = parsedBlockAddr & ~static_cast<uint64_t>(0xFULL);
    g_selectByBlockAddr = true;
    if (g_targetBlockAddr != parsedBlockAddr) {
      std::fprintf(
          stderr,
          "[singleblock16trace] aligned target block addr from %s to %s\n",
          blockAddrText.c_str(), hexAddr(g_targetBlockAddr).c_str());
    }
  }

  g_siteAllowlistFile = KnobSiteAllowlistFile.Value();
  if (!g_siteAllowlistFile.empty()) {
    if (!loadSiteAllowlist(g_siteAllowlistFile)) {
      std::fprintf(stderr, "failed to load -site-allowlist-file: %s\n",
                   g_siteAllowlistFile.c_str());
      return Usage();
    }
    g_selectBySiteAllowlist = true;
  }

  if (!g_selectByBlockAddr && !g_selectBySiteAllowlist) {
    std::fprintf(stderr,
                 "either -block-addr or -site-allowlist-file is required\n");
    return Usage();
  }

  g_taint_file = KnobTaintFile.Value();
  g_taint_no_lock = (KnobTaintNoLock.Value() != 0);
  g_seed_from_input_tensor = (KnobTaintSeedMode.Value() == "input-tensor");
  g_taint_enabled = (!g_taint_file.empty()) || g_seed_from_input_tensor;
  if (!g_taint_file.empty()) {
    g_taint_file_base = BaseName(g_taint_file);
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
