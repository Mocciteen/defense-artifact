#include "pin.H"

#include <cerrno>
#include <algorithm>
#include <array>
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
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

static KNOB<std::string> KnobOutput(KNOB_MODE_WRITEONCE, "pintool", "o",
                                    "site_bits.json",
                                    "Output JSON containing site-only bitstreams");
static KNOB<std::string> KnobAllowlist(
    KNOB_MODE_WRITEONCE, "pintool", "site-allowlist-file", "",
    "Path to a text file containing one target write site per line, e.g. "
    "libjit_conv2d_f+0x1a93");
static KNOB<std::string> KnobCallerIp(
    KNOB_MODE_WRITEONCE, "pintool", "caller-ip", "",
    "Optional exact immediate caller callsite IP filter. Only writes whose "
    "current callstack top matches this callsite are kept.");
static KNOB<std::string> KnobStackContainsIp(
    KNOB_MODE_WRITEONCE, "pintool", "stack-contains-ip", "",
    "Optional callsite IP filter. Only writes whose current callstack "
    "contains this callsite anywhere are kept.");
static KNOB<BOOL> KnobAddr4PrevWrite(
    KNOB_MODE_WRITEONCE, "pintool", "addr4-prev-write", "0",
    "If 1, emit per-4B-address streams instead of per-(site,vaddr,size) "
    "streams. Each bit compares the current 4B payload against the previous "
    "payload at the same 4B address; the first bit compares against the "
    "pre-write payload immediately before the first observed write.");

static PIN_LOCK g_stream_lock;
static TLS_KEY g_tls_key = INVALID_TLS_KEY;
static bool g_has_caller_ip_filter = false;
static uint64_t g_caller_ip_filter = 0;
static bool g_has_stack_contains_ip_filter = false;
static uint64_t g_stack_contains_ip_filter = 0;

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

struct SavedWriteSlot {
  bool valid = false;
  ADDRINT ea = 0;
  UINT32 size = 0;
  std::vector<uint8_t> before;
  struct SavedBlock {
    uint64_t blockAddr = 0;
    std::array<uint8_t, 16> before{};
  };
  std::vector<SavedBlock> blocks;
};

struct ThreadData {
  std::array<SavedWriteSlot, kMaxSavedMemOps> saved;
  std::vector<uint8_t> scratch;
  std::vector<uint64_t> callstack;
};

struct StreamKey {
  std::string site;
  uint64_t vaddr = 0;
  uint32_t size = 0;

  bool operator<(const StreamKey &other) const {
    if (site != other.site) {
      return site < other.site;
    }
    if (vaddr != other.vaddr) {
      return vaddr < other.vaddr;
    }
    return size < other.size;
  }
};

struct StreamState {
  SiteMeta meta;
  uint64_t vaddr = 0;
  uint32_t size = 0;
  uint64_t writeCount = 0;
  uint64_t unchangedCount = 0;
  uint64_t changedCount = 0;
  std::string bits;
};

struct WriterKey {
  std::string image;
  std::string imageOffset;
  std::string routine;
  std::string site;
  std::string ipHex;

  bool operator<(const WriterKey &other) const {
    if (image != other.image) {
      return image < other.image;
    }
    if (imageOffset != other.imageOffset) {
      return imageOffset < other.imageOffset;
    }
    if (routine != other.routine) {
      return routine < other.routine;
    }
    if (site != other.site) {
      return site < other.site;
    }
    return ipHex < other.ipHex;
  }
};

struct Addr4State {
  uint64_t vaddr = 0;
  uint64_t writeCount = 0;
  uint64_t unchangedCount = 0;
  uint64_t changedCount = 0;
  std::string bits;
  std::map<WriterKey, uint64_t> writerCounts;
  bool hasLast = false;
  std::array<uint8_t, 4> lastPayload{};
};

static std::unordered_set<std::string> g_allowlist;
static std::vector<std::string> g_allowlistOrder;
static std::map<uint64_t, SiteMeta> g_siteMetaByIp;
static std::set<std::string> g_matchedSites;
static std::set<std::string> g_matchedAllowlistEntries;
static std::set<std::string> g_sitesWithoutAfter;
static std::map<StreamKey, StreamState> g_streams;
static std::map<uint64_t, Addr4State> g_addr4Streams;

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

static std::string trim(const std::string &input) {
  size_t begin = 0;
  while (begin < input.size() &&
         (input[begin] == ' ' || input[begin] == '\t' || input[begin] == '\r' ||
          input[begin] == '\n')) {
    ++begin;
  }
  size_t end = input.size();
  while (end > begin &&
         (input[end - 1] == ' ' || input[end - 1] == '\t' ||
          input[end - 1] == '\r' || input[end - 1] == '\n')) {
    --end;
  }
  return input.substr(begin, end - begin);
}

static std::string stripComment(const std::string &line) {
  const size_t pos = line.find('#');
  return pos == std::string::npos ? line : line.substr(0, pos);
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

static WriterKey writerKeyFromMeta(const SiteMeta &meta) {
  WriterKey key;
  key.image = meta.image;
  key.imageOffset = meta.imageOffset;
  key.routine = meta.routine;
  key.site = meta.site;
  key.ipHex = meta.ipHex;
  return key;
}

static std::vector<std::pair<WriterKey, uint64_t>>
sortedWriterCounts(const std::map<WriterKey, uint64_t> &writerCounts) {
  std::vector<std::pair<WriterKey, uint64_t>> items(writerCounts.begin(),
                                                    writerCounts.end());
  std::sort(items.begin(), items.end(),
            [](const std::pair<WriterKey, uint64_t> &a,
               const std::pair<WriterKey, uint64_t> &b) {
              if (a.second != b.second) {
                return a.second > b.second;
              }
              if (a.first.image != b.first.image) {
                return a.first.image < b.first.image;
              }
              if (a.first.routine != b.first.routine) {
                return a.first.routine < b.first.routine;
              }
              if (a.first.site != b.first.site) {
                return a.first.site < b.first.site;
              }
              return a.first.ipHex < b.first.ipHex;
            });
  return items;
}

static void writeWriterObject(std::ofstream &out, const WriterKey &writer,
                              uint64_t writeCount) {
  out << "{";
  out << "\"image\": \"" << jsonEscape(writer.image) << "\", ";
  out << "\"image_offset\": \"" << jsonEscape(writer.imageOffset) << "\", ";
  out << "\"routine\": \"" << jsonEscape(writer.routine) << "\", ";
  out << "\"site\": \"" << jsonEscape(writer.site) << "\", ";
  out << "\"ip_hex\": \"" << jsonEscape(writer.ipHex) << "\", ";
  out << "\"write_count\": " << writeCount;
  out << "}";
}

static INT32 Usage() {
  std::fprintf(
      stderr,
      "sitebittrace: emit per-site 01 streams for a small write-site allowlist\n"
      "  -site-allowlist-file <path>  one target site per line (required)\n"
      "  -o <file>                    output JSON (default: site_bits.json)\n"
      "  -caller-ip <addr>            optional exact immediate caller callsite "
      "IP filter\n"
      "  -stack-contains-ip <addr>    optional callstack-membership filter\n"
      "  -addr4-prev-write 1          switch to per-4B-address previous-write "
      "comparison mode\n");
  return -1;
}

static void loadAllowlistOrExit() {
  const std::string path = KnobAllowlist.Value();
  if (path.empty()) {
    std::fprintf(stderr, "site-allowlist-file is required.\n");
    std::exit(2);
  }

  std::ifstream stream(path);
  if (!stream) {
    std::fprintf(stderr, "failed to open allowlist: %s\n", path.c_str());
    std::exit(2);
  }

  std::string raw;
  while (std::getline(stream, raw)) {
    const std::string line = trim(stripComment(raw));
    if (line.empty()) {
      continue;
    }
    if (g_allowlist.insert(line).second) {
      g_allowlistOrder.push_back(line);
    }
  }

  if (g_allowlist.empty()) {
    std::fprintf(stderr, "allowlist is empty: %s\n", path.c_str());
    std::exit(2);
  }
}

static std::string shortAliasForRoutine(const std::string &routine) {
  const size_t pos = routine.find("libjit_max_pool_generic");
  if (pos != std::string::npos) {
    return "libjit_max_pool_generic";
  }
  return "";
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
    const uint64_t imageOffset =
        ip - static_cast<uint64_t>(IMG_LowAddress(img));
    meta.imageOffset = offsetString(imageOffset);
  }

  const RTN rtn = RTN_FindByAddress(INS_Address(ins));
  if (RTN_Valid(rtn)) {
    meta.routine = RTN_Name(rtn);
    const uint64_t rtnOffset =
        ip - static_cast<uint64_t>(RTN_Address(rtn));
    meta.site = meta.routine + offsetString(rtnOffset);
  }

  return meta;
}

static bool siteIsAllowlisted(const SiteMeta &meta) {
  if (g_allowlist.find(meta.site) != g_allowlist.end()) {
    return true;
  }
  const std::string alias = shortAliasForRoutine(meta.routine);
  if (alias.empty()) {
    return false;
  }
  const std::string aliasSite =
      alias + meta.site.substr(meta.routine.size());
  return g_allowlist.find(aliasSite) != g_allowlist.end();
}

static std::vector<std::string> matchedAllowlistEntries(const SiteMeta &meta) {
  std::vector<std::string> matches;
  if (g_allowlist.find(meta.site) != g_allowlist.end()) {
    matches.push_back(meta.site);
  }

  const std::string alias = shortAliasForRoutine(meta.routine);
  if (!alias.empty()) {
    const std::string aliasSite =
        alias + meta.site.substr(meta.routine.size());
    if (g_allowlist.find(aliasSite) != g_allowlist.end() &&
        aliasSite != meta.site) {
      matches.push_back(aliasSite);
    }
  }

  return matches;
}

static VOID saveWriteBefore(THREADID tid, UINT32 memOp, ADDRINT ea,
                            UINT32 size) {
  if (size == 0 || memOp >= kMaxSavedMemOps) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  SavedWriteSlot &slot = td->saved[memOp];
  slot.valid = true;
  slot.ea = ea;
  slot.size = size;
  slot.before.resize(size);
  slot.blocks.clear();
  const size_t copied = PIN_SafeCopy(slot.before.data(),
                                     reinterpret_cast<const VOID *>(ea), size);
  if (copied < size) {
    std::memset(slot.before.data() + copied, 0, size - copied);
  }

  const uint64_t vaddr = static_cast<uint64_t>(ea);
  const uint64_t end = vaddr + static_cast<uint64_t>(size);
  const uint64_t start = vaddr & ~0xFULL;
  const uint64_t last = (end - 1) & ~0xFULL;
  for (uint64_t cur = start; cur <= last; cur += 16) {
    SavedWriteSlot::SavedBlock block;
    block.blockAddr = cur;
    const size_t blockCopied = PIN_SafeCopy(
        block.before.data(), reinterpret_cast<const VOID *>(cur), 16);
    if (blockCopied < 16) {
      std::memset(block.before.data() + blockCopied, 0, 16 - blockCopied);
    }
    slot.blocks.push_back(block);
  }
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

static bool CallstackContainsIp(const ThreadData *td, uint64_t ip) {
  if (td == nullptr) {
    return false;
  }
  for (uint64_t frameIp : td->callstack) {
    if (frameIp == ip) {
      return true;
    }
  }
  return false;
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
  if (!slot.valid || slot.size == 0) {
    return;
  }

  td->scratch.resize(slot.size);
  const size_t copied = PIN_SafeCopy(td->scratch.data(),
                                     reinterpret_cast<const VOID *>(slot.ea),
                                     slot.size);
  if (copied < slot.size) {
    std::memset(td->scratch.data() + copied, 0, slot.size - copied);
  }

  auto metaIt = g_siteMetaByIp.find(static_cast<uint64_t>(ip));
  if (metaIt != g_siteMetaByIp.end()) {
    const SiteMeta &meta = metaIt->second;
    const uint64_t callerIp = CurrentCallerIp(td);
    if (g_has_caller_ip_filter && callerIp != g_caller_ip_filter) {
      slot.valid = false;
      return;
    }
    if (g_has_stack_contains_ip_filter &&
        !CallstackContainsIp(td, g_stack_contains_ip_filter)) {
      slot.valid = false;
      return;
    }
    const uint64_t vaddr = static_cast<uint64_t>(slot.ea);
    const uint64_t end = vaddr + static_cast<uint64_t>(slot.size);

    if (KnobAddr4PrevWrite.Value()) {
      const WriterKey writer = writerKeyFromMeta(meta);
      const uint64_t firstLane = vaddr & ~0x3ULL;
      const uint64_t lastLane = (end - 1) & ~0x3ULL;

      PIN_GetLock(&g_stream_lock, tid + 1);
      for (uint64_t lane = firstLane; lane <= lastLane; lane += 4) {
        const uint64_t laneEnd = lane + 4;
        const uint64_t overlapStart = std::max(lane, vaddr);
        const uint64_t overlapEnd = std::min(laneEnd, end);
        if (overlapEnd <= overlapStart) {
          continue;
        }

        if (overlapStart != lane || overlapEnd != laneEnd) {
          continue;
        }

        const size_t offset = static_cast<size_t>(lane - vaddr);
        std::array<uint8_t, 4> beforeLane{};
        std::array<uint8_t, 4> afterLane{};
        std::memcpy(beforeLane.data(), slot.before.data() + offset, 4);
        std::memcpy(afterLane.data(), td->scratch.data() + offset, 4);

        Addr4State &state = g_addr4Streams[lane];
        if (state.bits.empty() && state.writeCount == 0) {
          state.vaddr = lane;
        }

        const uint8_t *compare = state.hasLast ? state.lastPayload.data()
                                               : beforeLane.data();
        const bool changed = std::memcmp(compare, afterLane.data(), 4) != 0;
        state.writeCount += 1;
        if (changed) {
          state.changedCount += 1;
          state.bits.push_back('1');
        } else {
          state.unchangedCount += 1;
          state.bits.push_back('0');
        }
        state.lastPayload = afterLane;
        state.hasLast = true;
        state.writerCounts[writer] += 1;
      }
      PIN_ReleaseLock(&g_stream_lock);
      slot.valid = false;
      return;
    }

    PIN_GetLock(&g_stream_lock, tid + 1);
    for (const SavedWriteSlot::SavedBlock &block : slot.blocks) {
      std::array<uint8_t, 16> afterBlock{};
      const size_t blockCopied =
          PIN_SafeCopy(afterBlock.data(),
                       reinterpret_cast<const VOID *>(block.blockAddr), 16);
      if (blockCopied < 16) {
        std::memset(afterBlock.data() + blockCopied, 0, 16 - blockCopied);
      }

      const bool changed =
          std::memcmp(block.before.data(), afterBlock.data(), 16) != 0;

      const StreamKey key{meta.site, block.blockAddr, 16};
      StreamState &state = g_streams[key];
      if (state.bits.empty() && state.writeCount == 0) {
        state.meta = meta;
        state.vaddr = block.blockAddr;
        state.size = 16;
      }
      state.writeCount += 1;
      if (changed) {
        state.changedCount += 1;
        state.bits.push_back('1');
      } else {
        state.unchangedCount += 1;
        state.bits.push_back('0');
      }
    }
    PIN_ReleaseLock(&g_stream_lock);
  }

  slot.valid = false;
}

static VOID ThreadStart(THREADID tid, CONTEXT *, INT32, VOID *) {
  ThreadData *td = new ThreadData();
  PIN_SetThreadData(g_tls_key, td, tid);
}

static VOID ThreadFini(THREADID tid, const CONTEXT *, INT32, VOID *) {
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  delete td;
  PIN_SetThreadData(g_tls_key, nullptr, tid);
}

static VOID Instruction(INS ins, VOID *) {
  if (INS_IsCall(ins)) {
    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(OnCall), IARG_THREAD_ID,
                   IARG_ADDRINT, INS_Address(ins), IARG_END);
  } else if (INS_IsRet(ins)) {
    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(OnRet), IARG_THREAD_ID,
                   IARG_END);
  }

  const UINT32 memOps = INS_MemoryOperandCount(ins);
  if (memOps == 0) {
    return;
  }

  bool hasWrite = false;
  for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
    if (INS_MemoryOperandIsWritten(ins, memOp)) {
      hasWrite = true;
      break;
    }
  }
  if (!hasWrite) {
    return;
  }

  const SiteMeta meta = describeSite(ins);
  if (!siteIsAllowlisted(meta)) {
    return;
  }

  g_siteMetaByIp.emplace(meta.ip, meta);
  g_matchedSites.insert(meta.site);
  for (const std::string &entry : matchedAllowlistEntries(meta)) {
    g_matchedAllowlistEntries.insert(entry);
  }

  const BOOL hasAfter = INS_IsValidForIpointAfter(ins);
  if (!hasAfter) {
    g_sitesWithoutAfter.insert(meta.site);
    return;
  }

  for (UINT32 memOp = 0; memOp < memOps; ++memOp) {
    if (!INS_MemoryOperandIsWritten(ins, memOp)) {
      continue;
    }
    INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(saveWriteBefore),
                             IARG_THREAD_ID, IARG_UINT32, memOp,
                             IARG_MEMORYOP_EA, memOp, IARG_MEMORYOP_SIZE,
                             memOp, IARG_END);
    INS_InsertPredicatedCall(ins, IPOINT_AFTER, AFUNPTR(recordWriteAfter),
                             IARG_THREAD_ID, IARG_UINT32, memOp, IARG_INST_PTR,
                             IARG_END);
  }
}

static void writeJsonArray(std::ofstream &out,
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

static VOID Fini(INT32, VOID *) {
  std::ofstream out(KnobOutput.Value().c_str());
  if (!out) {
    std::fprintf(stderr, "failed to open output: %s\n",
                 KnobOutput.Value().c_str());
    return;
  }

  std::vector<std::string> unmatchedSites;
  for (const std::string &site : g_allowlistOrder) {
    if (g_matchedAllowlistEntries.find(site) == g_matchedAllowlistEntries.end()) {
      unmatchedSites.push_back(site);
    }
  }

  std::vector<std::string> sitesWithoutAfter(g_sitesWithoutAfter.begin(),
                                             g_sitesWithoutAfter.end());

  out << "{\n";
  out << "  \"allowlist_file\": \"" << jsonEscape(KnobAllowlist.Value())
      << "\",\n";
  out << "  \"allowlist_count\": " << g_allowlistOrder.size() << ",\n";
  out << "  \"matched_site_count\": " << g_matchedAllowlistEntries.size()
      << ",\n";
  out << "  \"stream_count\": "
      << (KnobAddr4PrevWrite.Value() ? g_addr4Streams.size() : g_streams.size())
      << ",\n";
  out << "  \"caller_ip_filter\": "
      << (g_has_caller_ip_filter ? std::to_string(g_caller_ip_filter) : "null")
      << ",\n";
  out << "  \"caller_ip_filter_hex\": "
      << (g_has_caller_ip_filter ? ("\"" + hexAddr(g_caller_ip_filter) + "\"")
                                 : "null")
      << ",\n";
  out << "  \"stack_contains_ip_filter\": "
      << (g_has_stack_contains_ip_filter
              ? std::to_string(g_stack_contains_ip_filter)
              : "null")
      << ",\n";
  out << "  \"stack_contains_ip_filter_hex\": "
      << (g_has_stack_contains_ip_filter
              ? ("\"" + hexAddr(g_stack_contains_ip_filter) + "\"")
              : "null")
      << ",\n";
  out << "  \"unmatched_allowlist_sites\": ";
  writeJsonArray(out, unmatchedSites);
  out << ",\n";
  out << "  \"sites_without_after\": ";
  writeJsonArray(out, sitesWithoutAfter);
  out << ",\n";
  out << "  \"bit_definition\": ";
  if (KnobAddr4PrevWrite.Value()) {
    out << "\"Per-4B-address time-ordered 01 string over allowlisted write "
           "sites. Each bit compares the current 4B payload against the "
           "previous payload at the same 4B address; for the first observed "
           "write at that address, the comparison is against the pre-write "
           "payload immediately before that write. 0 means unchanged, 1 means "
           "changed.\",\n";
  } else {
    out << "\"Per-stream 01 string over allowlisted write sites. "
           "Each bit compares the full 16B-aligned block at the same "
           "(site, block_vaddr) stream immediately before and immediately "
           "after that write: "
           "0 means unchanged, 1 means changed.\",\n";
  }
  out << "  \"streams\": [\n";

  bool first = true;
  out << std::setprecision(17);
  if (KnobAddr4PrevWrite.Value()) {
    for (const auto &kv : g_addr4Streams) {
      const Addr4State &state = kv.second;
      const double unchangedRatio =
          state.writeCount == 0
              ? 0.0
              : static_cast<double>(state.unchangedCount) /
                    static_cast<double>(state.writeCount);
      const auto writers = sortedWriterCounts(state.writerCounts);

      if (!first) {
        out << ",\n";
      }
      first = false;
      out << "    {\n";
      out << "      \"vaddr\": " << state.vaddr << ",\n";
      out << "      \"vaddr_hex\": \"" << hexAddr(state.vaddr) << "\",\n";
      out << "      \"size\": 4,\n";
      out << "      \"write_count\": " << state.writeCount << ",\n";
      out << "      \"bits_len\": " << state.bits.size() << ",\n";
      out << "      \"unchanged_count\": " << state.unchangedCount << ",\n";
      out << "      \"changed_count\": " << state.changedCount << ",\n";
      out << "      \"unchanged_ratio\": " << unchangedRatio << ",\n";
      if (!writers.empty()) {
        out << "      \"owner\": ";
        writeWriterObject(out, writers[0].first, writers[0].second);
        out << ",\n";
      } else {
        out << "      \"owner\": null,\n";
      }
      out << "      \"owner_candidate_count\": " << writers.size() << ",\n";
      out << "      \"bits\": \"" << state.bits << "\"\n";
      out << "    }";
    }
  } else {
    for (const auto &kv : g_streams) {
      const StreamState &state = kv.second;
      const double unchangedRatio =
          state.writeCount == 0
              ? 0.0
              : static_cast<double>(state.unchangedCount) /
                    static_cast<double>(state.writeCount);

      if (!first) {
        out << ",\n";
      }
      first = false;
      out << "    {\n";
      out << "      \"site\": \"" << jsonEscape(state.meta.site) << "\",\n";
      out << "      \"ip_hex\": \"" << jsonEscape(state.meta.ipHex) << "\",\n";
      out << "      \"image\": \"" << jsonEscape(state.meta.image) << "\",\n";
      out << "      \"image_offset\": \"" << jsonEscape(state.meta.imageOffset)
          << "\",\n";
      out << "      \"routine\": \"" << jsonEscape(state.meta.routine)
          << "\",\n";
      out << "      \"disasm\": \"" << jsonEscape(state.meta.disasm)
          << "\",\n";
      out << "      \"vaddr\": " << state.vaddr << ",\n";
      out << "      \"vaddr_hex\": \"" << hexAddr(state.vaddr) << "\",\n";
      out << "      \"size\": " << state.size << ",\n";
      out << "      \"write_count\": " << state.writeCount << ",\n";
      out << "      \"compare_count\": " << state.writeCount << ",\n";
      out << "      \"bits_len\": " << state.bits.size() << ",\n";
      out << "      \"unchanged_count\": " << state.unchangedCount << ",\n";
      out << "      \"changed_count\": " << state.changedCount << ",\n";
      out << "      \"unchanged_ratio\": " << unchangedRatio << ",\n";
      out << "      \"bits\": \"" << state.bits << "\"\n";
      out << "    }";
    }
  }

  out << "\n  ]\n";
  out << "}\n";
  out.flush();
}

} // namespace

int main(int argc, char *argv[]) {
  PIN_InitSymbols();
  if (PIN_Init(argc, argv)) {
    return Usage();
  }

  if (!KnobCallerIp.Value().empty()) {
    if (!parseUint64Arg(KnobCallerIp.Value(), &g_caller_ip_filter)) {
      std::fprintf(stderr, "invalid -caller-ip: %s\n",
                   KnobCallerIp.Value().c_str());
      return 1;
    }
    g_has_caller_ip_filter = true;
  }
  if (!KnobStackContainsIp.Value().empty()) {
    if (!parseUint64Arg(KnobStackContainsIp.Value(),
                        &g_stack_contains_ip_filter)) {
      std::fprintf(stderr, "invalid -stack-contains-ip: %s\n",
                   KnobStackContainsIp.Value().c_str());
      return 1;
    }
    g_has_stack_contains_ip_filter = true;
  }

  loadAllowlistOrExit();

  PIN_InitLock(&g_stream_lock);
  g_tls_key = PIN_CreateThreadDataKey(nullptr);

  INS_AddInstrumentFunction(Instruction, nullptr);
  PIN_AddThreadStartFunction(ThreadStart, nullptr);
  PIN_AddThreadFiniFunction(ThreadFini, nullptr);
  PIN_AddFiniFunction(Fini, nullptr);

  PIN_StartProgram();
  return 0;
}
