#include "pin.H"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

static KNOB<std::string> KnobOutput(
    KNOB_MODE_WRITEONCE, "pintool", "o", "watchaddrtrace.tsv",
    "Output TSV containing watched writes with before/after values");
static KNOB<std::string> KnobIpMap(
    KNOB_MODE_WRITEONCE, "pintool", "m", "watchaddrtrace.ip.txt",
    "Instruction IP map output");
static KNOB<BOOL> KnobAlwaysOn(
    KNOB_MODE_WRITEONCE, "pintool", "always-on", "0",
    "Record writes without requiring pin_trace_begin/pin_trace_end.");
static KNOB<std::string> KnobWatchIps(
    KNOB_MODE_WRITEONCE, "pintool", "watch-ips", "",
    "Comma-separated instruction IPs to record, e.g. 0x401000,0x402000.");
static KNOB<std::string> KnobRuntimeWatchIps(
    KNOB_MODE_WRITEONCE, "pintool", "runtime-watch-ips", "",
    "Comma-separated instruction IPs to keep at runtime after routine-based "
    "instrumentation.");
static KNOB<std::string> KnobWatchRoutineSubstr(
    KNOB_MODE_WRITEONCE, "pintool", "watch-routine-substr", "",
    "Instrument memory writes whose routine name contains this substring.");
static KNOB<std::string> KnobWatchRanges(
    KNOB_MODE_WRITEONCE, "pintool", "watch-ranges", "",
    "Comma-separated watched ranges. Each item is start:size or start-end, "
    "e.g. 0x7fffffffca20:0x10 or 0x7fffffffca20-0x7fffffffca30.");
static KNOB<std::string> KnobStackContainsIp(
    KNOB_MODE_WRITEONCE, "pintool", "stack-contains-ip", "",
    "Optional callsite IP filter. Only writes whose current callstack "
    "contains this callsite anywhere are kept.");
static KNOB<BOOL> KnobOnlyUnchanged(
    KNOB_MODE_WRITEONCE, "pintool", "only-unchanged", "0",
    "Only emit writes whose before/after bytes are identical.");
static KNOB<BOOL> KnobCompact(
    KNOB_MODE_WRITEONCE, "pintool", "compact", "0",
    "Emit compact TSV without before/after hex payloads.");

constexpr size_t kMaxSavedMemOps = 8;

struct WatchRange {
  uint64_t start = 0;
  uint64_t end = 0;
};

struct SavedChunk {
  uint64_t vaddr = 0;
  uint32_t size = 0;
  uint32_t before_read = 0;
  std::vector<uint8_t> before;
};

struct SavedWriteSlot {
  bool valid = false;
  std::vector<SavedChunk> chunks;
};

struct ThreadData {
  std::array<SavedWriteSlot, kMaxSavedMemOps> saved{};
  std::vector<uint8_t> after;
  std::vector<uint64_t> callstack;
};

static FILE *g_out = nullptr;
static PIN_LOCK g_lock;
static PIN_LOCK g_inst_lock;
static std::atomic<uint64_t> g_seq{0};
static std::atomic<uint32_t> g_trace_scope_depth{0};
static TLS_KEY g_tls_key = INVALID_TLS_KEY;
static std::vector<WatchRange> g_watch_ranges;
static std::map<uint64_t, std::string> g_inst_map;
static std::unordered_set<uint64_t> g_watch_ips;
static std::unordered_set<uint64_t> g_runtime_watch_ips;
static bool g_has_stack_contains_ip = false;
static uint64_t g_stack_contains_ip = 0;

static inline bool TraceActive() {
  if (KnobAlwaysOn.Value()) {
    return true;
  }
  return g_trace_scope_depth.load(std::memory_order_acquire) != 0;
}

static bool ParseUint64(const std::string &text, uint64_t *out) {
  if (out == nullptr || text.empty()) {
    return false;
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 0);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  *out = static_cast<uint64_t>(value);
  return true;
}

static bool ParseWatchIps(const std::string &spec) {
  if (spec.empty()) {
    return true;
  }
  size_t pos = 0;
  while (pos < spec.size()) {
    const size_t next = spec.find(',', pos);
    const std::string item =
        spec.substr(pos, next == std::string::npos ? std::string::npos
                                                   : next - pos);
    uint64_t ip = 0;
    if (!ParseUint64(item, &ip)) {
      return false;
    }
    g_watch_ips.insert(ip);
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return true;
}

static bool ParseWatchRanges(const std::string &spec) {
  if (spec.empty()) {
    return true;
  }
  size_t pos = 0;
  while (pos < spec.size()) {
    const size_t next = spec.find(',', pos);
    const std::string item =
        spec.substr(pos, next == std::string::npos ? std::string::npos
                                                   : next - pos);
    const size_t colon = item.find(':');
    const size_t dash = item.find('-');
    uint64_t start = 0;
    uint64_t end = 0;
    if (colon != std::string::npos) {
      uint64_t size = 0;
      if (!ParseUint64(item.substr(0, colon), &start) ||
          !ParseUint64(item.substr(colon + 1), &size) || size == 0) {
        return false;
      }
      end = start + size;
      if (end <= start) {
        return false;
      }
    } else if (dash != std::string::npos) {
      if (!ParseUint64(item.substr(0, dash), &start) ||
          !ParseUint64(item.substr(dash + 1), &end) || end <= start) {
        return false;
      }
    } else {
      return false;
    }
    g_watch_ranges.push_back({start, end});
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return true;
}

static inline bool IpWatched(uint64_t ip) {
  return g_watch_ips.empty() || g_watch_ips.count(ip) != 0;
}

static inline bool RuntimeIpWatched(uint64_t ip) {
  return g_runtime_watch_ips.empty() || g_runtime_watch_ips.count(ip) != 0;
}

static bool ParseRuntimeWatchIps(const std::string &spec) {
  if (spec.empty()) {
    return true;
  }
  size_t pos = 0;
  while (pos < spec.size()) {
    const size_t next = spec.find(',', pos);
    const std::string item =
        spec.substr(pos, next == std::string::npos ? std::string::npos
                                                   : next - pos);
    uint64_t ip = 0;
    if (!ParseUint64(item, &ip)) {
      return false;
    }
    g_runtime_watch_ips.insert(ip);
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return true;
}

static VOID OnPinTraceBegin() {
  g_trace_scope_depth.fetch_add(1, std::memory_order_acq_rel);
}

static VOID OnPinTraceEnd() {
  const uint32_t depth = g_trace_scope_depth.load(std::memory_order_acquire);
  if (depth != 0) {
    g_trace_scope_depth.fetch_sub(1, std::memory_order_acq_rel);
  }
}

static VOID OnPinTraceClearWatches() { g_watch_ranges.clear(); }

static VOID OnPinTraceWatch(ADDRINT addr, ADDRINT size) {
  if (size == 0) {
    return;
  }
  const uint64_t start = static_cast<uint64_t>(addr);
  const uint64_t span = static_cast<uint64_t>(size);
  const uint64_t end = start + span;
  if (end <= start) {
    return;
  }
  g_watch_ranges.push_back({start, end});
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

static bool CallstackContainsIp(const ThreadData *td, uint64_t ip) {
  if (td == nullptr) {
    return false;
  }
  for (uint64_t frame_ip : td->callstack) {
    if (frame_ip == ip) {
      return true;
    }
  }
  return false;
}

static bool RuntimeWriteAllowed(const ThreadData *td, uint64_t ip) {
  if (!RuntimeIpWatched(ip)) {
    return false;
  }
  if (g_has_stack_contains_ip && !CallstackContainsIp(td, g_stack_contains_ip)) {
    return false;
  }
  return true;
}

static inline bool WatchOverlaps(uint64_t start, uint64_t size) {
  if (size == 0) {
    return false;
  }
  if (g_watch_ranges.empty()) {
    return true;
  }
  const uint64_t end = start + size;
  for (const WatchRange &range : g_watch_ranges) {
    if (start < range.end && range.start < end) {
      return true;
    }
  }
  return false;
}

static std::string BytesToHex(const uint8_t *data, uint32_t size) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize(static_cast<size_t>(size) * 2);
  for (uint32_t i = 0; i < size; ++i) {
    const uint8_t byte = data[i];
    out[static_cast<size_t>(i) * 2] = kHex[(byte >> 4) & 0xF];
    out[static_cast<size_t>(i) * 2 + 1] = kHex[byte & 0xF];
  }
  return out;
}

static void EmitRecord(THREADID tid, const char *kind, uint64_t vaddr,
                       uint32_t size, uint64_t ip, const char *changed,
                       uint32_t before_read, uint32_t after_read,
                       const uint8_t *before, const uint8_t *after) {
  if (KnobOnlyUnchanged.Value() && changed != nullptr &&
      std::strcmp(changed, "0") != 0) {
    return;
  }
  const uint64_t seq = g_seq.fetch_add(1, std::memory_order_relaxed);
  PIN_GetLock(&g_lock, tid + 1);
  if (KnobCompact.Value()) {
    std::fprintf(g_out,
                 "%" PRIu64 "\t%u\t%s\t0x%016" PRIx64 "\t%u\t0x%016" PRIx64
                 "\t%s\t%u\t%u\n",
                 seq, static_cast<uint32_t>(tid), kind, vaddr, size, ip,
                 changed, before_read, after_read);
  } else {
    const std::string before_hex =
        before != nullptr ? BytesToHex(before, size) : std::string("-");
    const std::string after_hex =
        after != nullptr ? BytesToHex(after, size) : std::string("-");
    std::fprintf(g_out,
                 "%" PRIu64 "\t%u\t%s\t0x%016" PRIx64 "\t%u\t0x%016" PRIx64
                 "\t%s\t%u\t%u\t%s\t%s\n",
                 seq, static_cast<uint32_t>(tid), kind, vaddr, size, ip,
                 changed, before_read, after_read, before_hex.c_str(),
                 after_hex.c_str());
  }
  PIN_ReleaseLock(&g_lock);
}

static VOID SaveWriteBefore(THREADID tid, UINT32 mem_op, ADDRINT addr,
                            UINT32 size, ADDRINT ip) {
  if (!TraceActive() || size == 0 || mem_op >= kMaxSavedMemOps) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  if (!RuntimeWriteAllowed(td, static_cast<uint64_t>(ip))) {
    return;
  }

  const uint64_t vaddr = static_cast<uint64_t>(addr);
  if (!WatchOverlaps(vaddr, static_cast<uint64_t>(size))) {
    return;
  }

  SavedWriteSlot &slot = td->saved[mem_op];
  slot.valid = false;
  slot.chunks.clear();

  const uint64_t end = vaddr + static_cast<uint64_t>(size);
  auto addChunk = [&](uint64_t overlap_start, uint64_t overlap_end) {
    SavedChunk chunk;
    chunk.vaddr = overlap_start;
    chunk.size = static_cast<uint32_t>(overlap_end - overlap_start);
    chunk.before.resize(chunk.size);
    const size_t copied = PIN_SafeCopy(
        chunk.before.data(), reinterpret_cast<const VOID *>(chunk.vaddr),
        chunk.size);
    chunk.before_read = static_cast<uint32_t>(copied);
    if (copied < chunk.size) {
      std::memset(chunk.before.data() + copied, 0, chunk.size - copied);
    }
    slot.chunks.push_back(std::move(chunk));
  };

  if (g_watch_ranges.empty()) {
    addChunk(vaddr, end);
  } else {
    for (const WatchRange &range : g_watch_ranges) {
      const uint64_t overlap_start = std::max(vaddr, range.start);
      const uint64_t overlap_end = std::min(end, range.end);
      if (overlap_end <= overlap_start) {
        continue;
      }
      addChunk(overlap_start, overlap_end);
    }
  }

  slot.valid = !slot.chunks.empty();
}

static VOID RecordWriteAfter(THREADID tid, UINT32 mem_op, ADDRINT ip) {
  if (!TraceActive() || g_out == nullptr || mem_op >= kMaxSavedMemOps) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  SavedWriteSlot &slot = td->saved[mem_op];
  if (!slot.valid) {
    return;
  }

  for (SavedChunk &chunk : slot.chunks) {
    if (td->after.size() < chunk.size) {
      td->after.resize(chunk.size);
    }
    const size_t copied = PIN_SafeCopy(
        td->after.data(), reinterpret_cast<const VOID *>(chunk.vaddr),
        chunk.size);
    if (copied < chunk.size) {
      std::memset(td->after.data() + copied, 0, chunk.size - copied);
    }

    const bool changed =
        std::memcmp(chunk.before.data(), td->after.data(), chunk.size) != 0;
    EmitRecord(tid, "after", chunk.vaddr, chunk.size, static_cast<uint64_t>(ip),
               changed ? "1" : "0", chunk.before_read,
               static_cast<uint32_t>(copied), chunk.before.data(),
               td->after.data());
  }

  slot.valid = false;
  slot.chunks.clear();
}

static VOID RecordWriteBeforeOnly(THREADID tid, ADDRINT addr, UINT32 size,
                                  ADDRINT ip) {
  if (!TraceActive() || g_out == nullptr || size == 0) {
    return;
  }
  ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }
  if (!RuntimeWriteAllowed(td, static_cast<uint64_t>(ip))) {
    return;
  }
  const uint64_t vaddr = static_cast<uint64_t>(addr);
  if (!WatchOverlaps(vaddr, static_cast<uint64_t>(size))) {
    return;
  }

  const uint64_t end = vaddr + static_cast<uint64_t>(size);
  std::vector<uint8_t> before;
  auto emitChunk = [&](uint64_t overlap_start, uint64_t overlap_end) {
    const uint32_t overlap_size =
        static_cast<uint32_t>(overlap_end - overlap_start);
    before.resize(overlap_size);
    const size_t copied = PIN_SafeCopy(
        before.data(), reinterpret_cast<const VOID *>(overlap_start),
        overlap_size);
    if (copied < overlap_size) {
      std::memset(before.data() + copied, 0, overlap_size - copied);
    }
    EmitRecord(tid, "before_only", overlap_start, overlap_size,
               static_cast<uint64_t>(ip), "NA",
               static_cast<uint32_t>(copied), 0, before.data(), nullptr);
  };

  if (g_watch_ranges.empty()) {
    emitChunk(vaddr, end);
  } else {
    for (const WatchRange &range : g_watch_ranges) {
      const uint64_t overlap_start = std::max(vaddr, range.start);
      const uint64_t overlap_end = std::min(end, range.end);
      if (overlap_end <= overlap_start) {
        continue;
      }
      emitChunk(overlap_start, overlap_end);
    }
  }
}

static VOID RegisterInstInfo(INS ins) {
  const uint64_t ip = static_cast<uint64_t>(INS_Address(ins));
  if (g_inst_map.find(ip) != g_inst_map.end()) {
    return;
  }

  std::string image_name = "<unknown>";
  uint64_t image_offset = 0;
  const IMG img = IMG_FindByAddress(INS_Address(ins));
  if (IMG_Valid(img)) {
    image_name = IMG_Name(img);
    image_offset = ip - static_cast<uint64_t>(IMG_LowAddress(img));
  }

  std::string rtn_name = "-";
  uint64_t rtn_offset = 0;
  const RTN rtn = RTN_FindByAddress(INS_Address(ins));
  if (RTN_Valid(rtn)) {
    rtn_name = RTN_Name(rtn);
    rtn_offset = ip - static_cast<uint64_t>(RTN_Address(rtn));
  }

  const std::string disasm = INS_Disassemble(ins);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "0x%016" PRIx64, ip);

  std::string line;
  line.reserve(256);
  line.append(buf);
  line.push_back('\t');
  line.append(image_name);
  line.push_back('\t');
  std::snprintf(buf, sizeof(buf), "+0x%" PRIx64, image_offset);
  line.append(buf);
  line.push_back('\t');
  if (rtn_name != "-") {
    line.append(rtn_name);
    std::snprintf(buf, sizeof(buf), "+0x%" PRIx64, rtn_offset);
    line.append(buf);
  } else {
    line.append("-");
  }
  line.push_back('\t');
  line.append(disasm);

  PIN_GetLock(&g_inst_lock, 1);
  if (g_inst_map.find(ip) == g_inst_map.end()) {
    g_inst_map[ip] = std::move(line);
  }
  PIN_ReleaseLock(&g_inst_lock);
}

static VOID Routine(RTN rtn, VOID *) {
  if (!RTN_Valid(rtn)) {
    return;
  }
  const std::string &name = RTN_Name(rtn);
  if (name == "pin_trace_begin") {
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnPinTraceBegin), IARG_END);
    RTN_Close(rtn);
    return;
  }
  if (name == "pin_trace_end") {
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnPinTraceEnd), IARG_END);
    RTN_Close(rtn);
    return;
  }
  if (name == "pin_trace_clear_watches") {
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnPinTraceClearWatches),
                   IARG_END);
    RTN_Close(rtn);
    return;
  }
  if (name == "pin_trace_watch") {
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnPinTraceWatch),
                   IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                   IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_END);
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
  if (INS_HasScatteredMemoryAccess(ins)) {
    return;
  }
  const uint64_t ip = static_cast<uint64_t>(INS_Address(ins));
  const std::string routine_substr = KnobWatchRoutineSubstr.Value();
  if (!routine_substr.empty()) {
    const RTN rtn = RTN_FindByAddress(INS_Address(ins));
    if (!RTN_Valid(rtn) ||
        RTN_Name(rtn).find(routine_substr) == std::string::npos) {
      return;
    }
  } else if (!IpWatched(ip)) {
    return;
  }
  const UINT32 mem_ops = INS_MemoryOperandCount(ins);
  if (mem_ops == 0) {
    return;
  }

  const BOOL has_after = INS_IsValidForIpointAfter(ins);
  for (UINT32 mem_op = 0; mem_op < mem_ops; ++mem_op) {
    if (!INS_MemoryOperandIsWritten(ins, mem_op)) {
      continue;
    }
    const UINT32 mem_size =
        static_cast<UINT32>(INS_MemoryOperandSize(ins, mem_op));
    if (mem_size == 0) {
      continue;
    }
    RegisterInstInfo(ins);
    if (has_after) {
      INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(SaveWriteBefore),
                               IARG_THREAD_ID, IARG_UINT32, mem_op,
                               IARG_MEMORYOP_EA, mem_op, IARG_UINT32, mem_size,
                               IARG_INST_PTR,
                               IARG_END);
      INS_InsertPredicatedCall(ins, IPOINT_AFTER, AFUNPTR(RecordWriteAfter),
                               IARG_THREAD_ID, IARG_UINT32, mem_op,
                               IARG_INST_PTR, IARG_END);
    } else {
      INS_InsertPredicatedCall(ins, IPOINT_BEFORE,
                               AFUNPTR(RecordWriteBeforeOnly), IARG_THREAD_ID,
                               IARG_MEMORYOP_EA, mem_op, IARG_UINT32, mem_size,
                               IARG_INST_PTR, IARG_END);
    }
  }
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

static VOID Fini(INT32, VOID *) {
  if (g_out != nullptr) {
    std::fflush(g_out);
    std::fclose(g_out);
    g_out = nullptr;
  }
  if (!g_inst_map.empty()) {
    FILE *map = std::fopen(KnobIpMap.Value().c_str(), "w");
    if (map != nullptr) {
      std::fprintf(map, "# ip\timage\timage_offset\troutine\tdisasm\n");
      for (const auto &kv : g_inst_map) {
        std::fprintf(map, "%s\n", kv.second.c_str());
      }
      std::fflush(map);
      std::fclose(map);
    }
  }
}

static INT32 Usage() {
  std::fprintf(stderr,
               "watchaddrtrace: record watched writes with before/after values "
               "inside scoped trace window\n");
  return -1;
}

} // namespace

int main(int argc, char *argv[]) {
  PIN_InitSymbols();
  if (PIN_Init(argc, argv)) {
    return Usage();
  }
  if (!ParseWatchIps(KnobWatchIps.Value())) {
    std::fprintf(stderr, "invalid -watch-ips: %s\n",
                 KnobWatchIps.Value().c_str());
    return 1;
  }
  if (!ParseRuntimeWatchIps(KnobRuntimeWatchIps.Value())) {
    std::fprintf(stderr, "invalid -runtime-watch-ips: %s\n",
                 KnobRuntimeWatchIps.Value().c_str());
    return 1;
  }
  if (!ParseWatchRanges(KnobWatchRanges.Value())) {
    std::fprintf(stderr, "invalid -watch-ranges: %s\n",
                 KnobWatchRanges.Value().c_str());
    return 1;
  }
  if (!KnobStackContainsIp.Value().empty()) {
    if (!ParseUint64(KnobStackContainsIp.Value(), &g_stack_contains_ip)) {
      std::fprintf(stderr, "invalid -stack-contains-ip: %s\n",
                   KnobStackContainsIp.Value().c_str());
      return 1;
    }
    g_has_stack_contains_ip = true;
  }

  g_out = std::fopen(KnobOutput.Value().c_str(), "w");
  if (g_out == nullptr) {
    std::perror("fopen trace output");
    return 1;
  }
  std::fprintf(g_out,
               "%s",
               KnobCompact.Value()
                   ? "# seq\ttid\tkind\tvaddr\tsize\tip\tchanged\tbefore_read\t"
                     "after_read\n"
                   : "# seq\ttid\tkind\tvaddr\tsize\tip\tchanged\tbefore_read\t"
                     "after_read\tbefore_hex\tafter_hex\n");
  std::fflush(g_out);

  PIN_InitLock(&g_lock);
  PIN_InitLock(&g_inst_lock);
  g_tls_key = PIN_CreateThreadDataKey(nullptr);

  RTN_AddInstrumentFunction(Routine, nullptr);
  INS_AddInstrumentFunction(Instruction, nullptr);
  PIN_AddThreadStartFunction(ThreadStart, nullptr);
  PIN_AddThreadFiniFunction(ThreadFini, nullptr);
  PIN_AddFiniFunction(Fini, nullptr);
  PIN_StartProgram();
  return 0;
}
