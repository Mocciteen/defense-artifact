#include "pintrace_common.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace {

using pintrace::RecordHeader;

KNOB<std::string> KnobOutput(KNOB_MODE_WRITEONCE, "pintool", "o",
                             "paddrtrace.bin", "Output trace file");
KNOB<std::string> KnobIpMap(KNOB_MODE_WRITEONCE, "pintool", "m",
                            "paddrtrace.ip.txt", "Instruction IP map output");
KNOB<UINT32> KnobStackDepth(KNOB_MODE_WRITEONCE, "pintool", "stack-depth", "0",
                            "Compatibility option. This artifact emits stack_depth=0.");
KNOB<BOOL> KnobAfterOnly(KNOB_MODE_WRITEONCE, "pintool", "after-only", "1",
                         "Emit post-write records. Kept for compatibility.");
KNOB<std::string> KnobSiteIp(KNOB_MODE_WRITEONCE, "pintool", "site-ip", "",
                             "Optional exact write-site IP filter.");

struct ThreadData {
  std::vector<uint8_t> write_buf;
  std::array<uint8_t, pintrace::kBlockSize> block16{};
  std::array<ADDRINT, 8> saved_ea{};
  std::array<UINT32, 8> saved_size{};
  std::array<uint8_t, 8> saved_valid{};
  std::unordered_map<uint64_t, uint64_t> pfn_cache;
};

PIN_LOCK g_file_lock;
PIN_LOCK g_ipmap_lock;
TLS_KEY g_tls_key = INVALID_TLS_KEY;
FILE *g_out = nullptr;
int g_pagemap_fd = -1;
std::atomic<uint64_t> g_seq{0};
std::map<uint64_t, pintrace::SiteMeta> g_ipmap;
bool g_has_site_filter = false;
uint64_t g_site_filter = 0;
uint64_t g_page_size = 4096;

uint64_t ResolvePhysicalAddress(uint64_t runtime_addr, ThreadData *td, bool *ok) {
  *ok = false;
  if (g_pagemap_fd < 0 || td == nullptr || g_page_size == 0) {
    return 0;
  }

  const uint64_t page_index = runtime_addr / g_page_size;
  const uint64_t page_offset = runtime_addr % g_page_size;
  const auto cached = td->pfn_cache.find(page_index);
  if (cached != td->pfn_cache.end()) {
    *ok = true;
    return cached->second + page_offset;
  }

  uint64_t entry = 0;
  const off_t offset = static_cast<off_t>(page_index * sizeof(entry));
  const ssize_t got = pread(g_pagemap_fd, &entry, sizeof(entry), offset);
  if (got != static_cast<ssize_t>(sizeof(entry))) {
    return 0;
  }
  const bool present = (entry >> 63) != 0;
  const uint64_t pfn = entry & ((1ULL << 55) - 1);
  if (!present || pfn == 0) {
    return 0;
  }

  const uint64_t page_paddr = pfn * g_page_size;
  td->pfn_cache[page_index] = page_paddr;
  *ok = true;
  return page_paddr + page_offset;
}

void SaveWriteEA(THREADID tid, UINT32 mem_op, ADDRINT ea, UINT32 size) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || mem_op >= td->saved_ea.size()) {
    return;
  }
  td->saved_ea[mem_op] = ea;
  td->saved_size[mem_op] = size;
  td->saved_valid[mem_op] = 1;
}

void EmitRecord(const RecordHeader &hdr, const uint8_t *payload,
                const uint8_t *block16) {
  PIN_GetLock(&g_file_lock, 1);
  std::fwrite(&hdr, sizeof(hdr), 1, g_out);
  if (hdr.size > 0) {
    std::fwrite(payload, 1, hdr.size, g_out);
  }
  std::fwrite(block16, 1, pintrace::kBlockSize, g_out);
  PIN_ReleaseLock(&g_file_lock);
}

void RecordWrite(THREADID tid, ADDRINT ea, UINT32 size, ADDRINT ip) {
  if (size == 0 || (g_has_site_filter && static_cast<uint64_t>(ip) != g_site_filter)) {
    return;
  }
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr) {
    return;
  }

  td->write_buf.resize(size);
  const size_t size_read = pintrace::SafeCopyZero(td->write_buf.data(), ea, size);
  const uint64_t start = static_cast<uint64_t>(ea);
  const uint64_t end = start + static_cast<uint64_t>(size);
  const uint64_t first_block = pintrace::AlignDown16(start);
  const uint64_t last_block = pintrace::AlignDown16(end + pintrace::kBlockSize - 1);

  for (uint64_t block = first_block; block < last_block; block += pintrace::kBlockSize) {
    const uint64_t overlap_start = std::max(block, start);
    const uint64_t overlap_end = std::min(block + pintrace::kBlockSize, end);
    if (overlap_end <= overlap_start) {
      continue;
    }
    const uint32_t overlap_len = static_cast<uint32_t>(overlap_end - overlap_start);
    const size_t offset = static_cast<size_t>(overlap_start - start);
    const uint32_t part_read = size_read > offset
                                   ? static_cast<uint32_t>(
                                         std::min<size_t>(overlap_len, size_read - offset))
                                   : 0;
    const size_t block_read =
        pintrace::SafeCopyZero(td->block16.data(), static_cast<ADDRINT>(block),
                               td->block16.size());
    bool ok_paddr = false;
    bool ok_paddr16 = false;
    const uint64_t paddr = ResolvePhysicalAddress(overlap_start, td, &ok_paddr);
    const uint64_t paddr16 = ResolvePhysicalAddress(block, td, &ok_paddr16);
    if (!ok_paddr || !ok_paddr16) {
      continue;
    }

    RecordHeader hdr{};
    hdr.seq = g_seq.fetch_add(1, std::memory_order_relaxed);
    hdr.tid = static_cast<uint32_t>(tid);
    hdr.size = overlap_len;
    hdr.size_read = part_read;
    hdr.block16_read = static_cast<uint32_t>(block_read);
    hdr.compat_addr = paddr;
    hdr.paddr = paddr;
    hdr.paddr16 = paddr16;
    hdr.ip = static_cast<uint64_t>(ip);
    hdr.instr_id = 0;
    hdr.flags = pintrace::kFlagPaddrValid | pintrace::kFlagPaddr16Valid;
    if (part_read != overlap_len) {
      hdr.flags |= pintrace::kFlagPartialWriteRead;
    }
    if (block_read != td->block16.size()) {
      hdr.flags |= pintrace::kFlagPartialBlock16Read;
    }
    hdr.stack_depth = 0;

    EmitRecord(hdr, td->write_buf.data() + offset, td->block16.data());
  }
}

void RecordSavedWrite(THREADID tid, UINT32 mem_op, ADDRINT ip) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || mem_op >= td->saved_valid.size() || !td->saved_valid[mem_op]) {
    return;
  }
  const ADDRINT ea = td->saved_ea[mem_op];
  const UINT32 size = td->saved_size[mem_op];
  td->saved_valid[mem_op] = 0;
  RecordWrite(tid, ea, size, ip);
}

void Instruction(INS ins, VOID *) {
  PIN_GetLock(&g_ipmap_lock, 1);
  pintrace::RememberInstruction(&g_ipmap, ins);
  PIN_ReleaseLock(&g_ipmap_lock);

  if (INS_HasScatteredMemoryAccess(ins) || INS_IsVscatter(ins) || INS_IsVgather(ins)) {
    return;
  }
  if (!INS_IsValidForIpointAfter(ins)) {
    return;
  }

  const UINT32 mem_ops = INS_MemoryOperandCount(ins);
  for (UINT32 mem_op = 0; mem_op < mem_ops && mem_op < 8; ++mem_op) {
    if (!INS_MemoryOperandIsWritten(ins, mem_op)) {
      continue;
    }
    const UINT32 size = static_cast<UINT32>(INS_MemoryOperandSize(ins, mem_op));
    if (size == 0) {
      continue;
    }
    INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(SaveWriteEA),
                             IARG_THREAD_ID, IARG_UINT32, mem_op,
                             IARG_MEMORYOP_EA, mem_op, IARG_UINT32, size,
                             IARG_END);
    INS_InsertPredicatedCall(ins, IPOINT_AFTER, AFUNPTR(RecordSavedWrite),
                             IARG_THREAD_ID, IARG_UINT32, mem_op,
                             IARG_INST_PTR, IARG_END);
  }
}

void ThreadStart(THREADID tid, CONTEXT *, INT32, VOID *) {
  auto *td = new ThreadData();
  PIN_SetThreadData(g_tls_key, td, tid);
}

void ThreadFini(THREADID tid, const CONTEXT *, INT32, VOID *) {
  delete static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  PIN_SetThreadData(g_tls_key, nullptr, tid);
}

void Fini(INT32, VOID *) {
  if (g_out != nullptr) {
    std::fflush(g_out);
    std::fclose(g_out);
    g_out = nullptr;
  }
  if (g_pagemap_fd >= 0) {
    close(g_pagemap_fd);
    g_pagemap_fd = -1;
  }
  pintrace::WriteIpMap(KnobIpMap.Value(), g_ipmap);
}

INT32 Usage() {
  std::fprintf(stderr,
               "paddrtrace: collect post-write PADDRTRC v4 records.\n"
               "  -o <file>           output binary trace\n"
               "  -m <file>           output IP map\n"
               "  -site-ip <hex>      optional exact write-site filter\n"
               "  -stack-depth 0      compatibility option\n");
  return -1;
}

}  // namespace

int main(int argc, char *argv[]) {
  PIN_InitSymbols();
  if (PIN_Init(argc, argv)) {
    return Usage();
  }
  if (!KnobSiteIp.Value().empty()) {
    if (!pintrace::ParseU64(KnobSiteIp.Value(), &g_site_filter)) {
      std::fprintf(stderr, "invalid -site-ip: %s\n", KnobSiteIp.Value().c_str());
      return 1;
    }
    g_has_site_filter = true;
  }
  if (KnobStackDepth.Value() != 0) {
    std::fprintf(stderr, "warning: artifact paddrtrace emits stack_depth=0\n");
  }
  if (!KnobAfterOnly.Value()) {
    std::fprintf(stderr, "warning: artifact paddrtrace is after-only\n");
  }

  g_page_size = static_cast<uint64_t>(::getpagesize());
  g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
  if (g_pagemap_fd < 0) {
    std::fprintf(stderr, "failed to open /proc/self/pagemap; physical address mode needs sufficient privileges\n");
    return 1;
  }
  g_out = std::fopen(KnobOutput.Value().c_str(), "wb");
  if (g_out == nullptr) {
    std::fprintf(stderr, "failed to open output: %s\n", KnobOutput.Value().c_str());
    return 1;
  }
  pintrace::WriteFileHeader(g_out, g_page_size);

  PIN_InitLock(&g_file_lock);
  PIN_InitLock(&g_ipmap_lock);
  g_tls_key = PIN_CreateThreadDataKey(nullptr);
  INS_AddInstrumentFunction(Instruction, nullptr);
  PIN_AddThreadStartFunction(ThreadStart, nullptr);
  PIN_AddThreadFiniFunction(ThreadFini, nullptr);
  PIN_AddFiniFunction(Fini, nullptr);
  PIN_StartProgram();
  return 0;
}
