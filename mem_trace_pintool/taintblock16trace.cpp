#include "pintrace_common.h"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace {

KNOB<std::string> KnobOutput(KNOB_MODE_WRITEONCE, "pintool", "o",
                             "taint_block16_bits.json", "Output JSON summary");
KNOB<std::string> KnobIpMap(KNOB_MODE_WRITEONCE, "pintool", "m", "",
                            "Optional IP map output");
KNOB<std::string> KnobTaintFile(KNOB_MODE_WRITEONCE, "pintool", "taint-file", "",
                                "Input file path or basename used as taint seed");
KNOB<std::string> KnobTaintSeedMode(KNOB_MODE_WRITEONCE, "pintool",
                                    "taint-seed-mode", "file",
                                    "Only file mode is kept in this artifact");
KNOB<BOOL> KnobTaintOnly(KNOB_MODE_WRITEONCE, "pintool", "taint-only", "1",
                         "Keep only writes whose value is tainted");
KNOB<UINT32> KnobStackDepth(KNOB_MODE_WRITEONCE, "pintool", "stack-depth", "0",
                            "Compatibility option. Stack capture is omitted.");
KNOB<std::string> KnobSiteIp(KNOB_MODE_WRITEONCE, "pintool", "site-ip", "",
                             "Optional exact write-site IP filter");
KNOB<BOOL> KnobSiteBitsOnly(KNOB_MODE_WRITEONCE, "pintool", "site-bits-only", "0",
                            "Emit only the selected site's 01 string");
KNOB<std::string> KnobSiteBitsText(KNOB_MODE_WRITEONCE, "pintool", "site-bits-txt",
                                  "", "Optional raw 01 output for site-bits-only");

struct ShadowPage {
  std::array<uint64_t, 64> bits{};
};

struct SavedBlock {
  uint64_t runtime_block = 0;
  uint64_t paddr_block = 0;
  std::array<uint8_t, pintrace::kBlockSize> before{};
};

struct SavedWrite {
  bool valid = false;
  bool tainted = false;
  ADDRINT addr = 0;
  UINT32 size = 0;
  std::vector<SavedBlock> blocks;
};

struct BlockState {
  uint64_t paddr = 0;
  uint64_t write_count = 0;
  uint64_t unchanged_count = 0;
  uint64_t changed_count = 0;
  std::string bits;
  std::map<uint64_t, uint64_t> owner_counts;
};

struct SiteState {
  uint64_t ip = 0;
  uint64_t write_count = 0;
  uint64_t unchanged_count = 0;
  uint64_t changed_count = 0;
  std::string bits;
};

struct ThreadData {
  static constexpr size_t kMaxMemOps = 8;
  bool cur_taint = false;
  std::vector<uint8_t> reg_taint;
  std::array<SavedWrite, kMaxMemOps> saved;
  std::array<uint8_t, pintrace::kBlockSize> after{};
  ADDRINT last_sys_num = 0;
  ADDRINT last_sys_arg0 = 0;
  ADDRINT last_sys_arg1 = 0;
  ADDRINT last_sys_arg2 = 0;
  ADDRINT last_sys_arg4 = 0;
  bool last_open_match = false;
  std::unordered_map<uint64_t, uint64_t> pfn_cache;
};

PIN_LOCK g_stream_lock;
PIN_LOCK g_taint_lock;
PIN_LOCK g_ipmap_lock;
TLS_KEY g_tls_key = INVALID_TLS_KEY;
std::unordered_map<uint64_t, ShadowPage *> g_shadow_pages;
std::unordered_set<int> g_taint_fds;
std::map<uint64_t, BlockState> g_blocks;
std::map<uint64_t, SiteState> g_sites;
std::map<uint64_t, pintrace::SiteMeta> g_ipmap;
std::string g_taint_file;
bool g_has_site_filter = false;
uint64_t g_site_filter = 0;
bool g_site_bits_only = false;
int g_pagemap_fd = -1;
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

ShadowPage *GetPage(uint64_t addr, bool create) {
  const uint64_t page = addr >> 12;
  auto it = g_shadow_pages.find(page);
  if (it != g_shadow_pages.end()) {
    return it->second;
  }
  if (!create) {
    return nullptr;
  }
  auto *shadow = new ShadowPage();
  g_shadow_pages[page] = shadow;
  return shadow;
}

void MemSetTaint(uint64_t addr, uint32_t size, bool tainted) {
  if (size == 0) {
    return;
  }
  PIN_GetLock(&g_taint_lock, 1);
  for (uint64_t cur = addr; cur < addr + size; ++cur) {
    ShadowPage *page = GetPage(cur, tainted);
    if (page == nullptr) {
      continue;
    }
    const uint64_t offset = cur & 0xFFFu;
    const uint64_t word = offset >> 6;
    const uint64_t bit = offset & 63u;
    const uint64_t mask = 1ULL << bit;
    if (tainted) {
      page->bits[word] |= mask;
    } else {
      page->bits[word] &= ~mask;
    }
  }
  PIN_ReleaseLock(&g_taint_lock);
}

bool MemAnyTaint(uint64_t addr, uint32_t size) {
  if (size == 0) {
    return false;
  }
  PIN_GetLock(&g_taint_lock, 1);
  for (uint64_t cur = addr; cur < addr + size; ++cur) {
    ShadowPage *page = GetPage(cur, false);
    if (page == nullptr) {
      continue;
    }
    const uint64_t offset = cur & 0xFFFu;
    if ((page->bits[offset >> 6] >> (offset & 63u)) & 1ULL) {
      PIN_ReleaseLock(&g_taint_lock);
      return true;
    }
  }
  PIN_ReleaseLock(&g_taint_lock);
  return false;
}

REG NormReg(REG reg) {
  if (reg == REG_INVALID()) {
    return reg;
  }
  return REG_FullRegName(reg);
}

bool GetRegTaint(ThreadData *td, REG reg) {
  reg = NormReg(reg);
  const size_t idx = static_cast<size_t>(reg);
  return reg != REG_INVALID() && idx < td->reg_taint.size() && td->reg_taint[idx] != 0;
}

void SetRegTaint(ThreadData *td, REG reg, bool tainted) {
  reg = NormReg(reg);
  const size_t idx = static_cast<size_t>(reg);
  if (reg != REG_INVALID() && idx < td->reg_taint.size()) {
    td->reg_taint[idx] = tainted ? 1 : 0;
  }
}

void TaintBegin(THREADID tid) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr) {
    td->cur_taint = false;
  }
}

void TaintAccReg(THREADID tid, UINT32 reg_id) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr && GetRegTaint(td, static_cast<REG>(reg_id))) {
    td->cur_taint = true;
  }
}

void TaintAccMem(THREADID tid, ADDRINT addr, UINT32 size) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr && MemAnyTaint(static_cast<uint64_t>(addr), size)) {
    td->cur_taint = true;
  }
}

void TaintSetReg(THREADID tid, UINT32 reg_id) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td != nullptr) {
    SetRegTaint(td, static_cast<REG>(reg_id), td->cur_taint);
  }
}

void SaveWriteBefore(THREADID tid, UINT32 mem_op, ADDRINT ea, UINT32 size,
                     ADDRINT ip) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || mem_op >= td->saved.size() || size == 0) {
    return;
  }
  SavedWrite &slot = td->saved[mem_op];
  slot.valid = true;
  slot.tainted = td->cur_taint;
  slot.addr = ea;
  slot.size = size;
  slot.blocks.clear();

  if (g_has_site_filter && static_cast<uint64_t>(ip) != g_site_filter) {
    slot.tainted = false;
    return;
  }
  if (KnobTaintOnly.Value() && !slot.tainted) {
    return;
  }

  const uint64_t start = static_cast<uint64_t>(ea);
  const uint64_t end = start + static_cast<uint64_t>(size);
  const uint64_t first_block = pintrace::AlignDown16(start);
  const uint64_t last_block = pintrace::AlignDown16(end + pintrace::kBlockSize - 1);
  for (uint64_t block = first_block; block < last_block; block += pintrace::kBlockSize) {
    bool ok_paddr = false;
    const uint64_t paddr = ResolvePhysicalAddress(block, td, &ok_paddr);
    if (!ok_paddr) {
      continue;
    }
    SavedBlock saved;
    saved.runtime_block = block;
    saved.paddr_block = paddr;
    pintrace::SafeCopyZero(saved.before.data(), static_cast<ADDRINT>(block),
                           saved.before.size());
    slot.blocks.push_back(saved);
  }
}

void UpdateSite(uint64_t ip, bool changed) {
  SiteState &site = g_sites[ip];
  if (site.write_count == 0) {
    site.ip = ip;
  }
  site.write_count += 1;
  if (changed) {
    site.changed_count += 1;
    site.bits.push_back('1');
  } else {
    site.unchanged_count += 1;
    site.bits.push_back('0');
  }
}

void CompareWriteAfter(THREADID tid, UINT32 mem_op, ADDRINT ip) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || mem_op >= td->saved.size()) {
    return;
  }
  SavedWrite &slot = td->saved[mem_op];
  if (!slot.valid) {
    return;
  }
  slot.valid = false;

  if (g_has_site_filter && static_cast<uint64_t>(ip) != g_site_filter) {
    return;
  }
  if (KnobTaintOnly.Value() && !slot.tainted) {
    return;
  }
  if (slot.tainted) {
    MemSetTaint(static_cast<uint64_t>(slot.addr), slot.size, true);
  }

  PIN_GetLock(&g_stream_lock, tid + 1);
  for (const SavedBlock &block : slot.blocks) {
    pintrace::SafeCopyZero(td->after.data(), static_cast<ADDRINT>(block.runtime_block),
                           td->after.size());
    const bool changed =
        std::memcmp(block.before.data(), td->after.data(), pintrace::kBlockSize) != 0;

    if (g_site_bits_only) {
      UpdateSite(static_cast<uint64_t>(ip), changed);
      continue;
    }

    BlockState &state = g_blocks[block.paddr_block];
    if (state.write_count == 0) {
      state.paddr = block.paddr_block;
    }
    state.write_count += 1;
    state.owner_counts[static_cast<uint64_t>(ip)] += 1;
    if (changed) {
      state.changed_count += 1;
      state.bits.push_back('1');
    } else {
      state.unchanged_count += 1;
      state.bits.push_back('0');
    }
    UpdateSite(static_cast<uint64_t>(ip), changed);
  }
  PIN_ReleaseLock(&g_stream_lock);
}

bool ReadUserString(ADDRINT ptr, std::string *out) {
  char buf[512];
  const size_t copied = PIN_SafeCopy(buf, reinterpret_cast<const VOID *>(ptr), sizeof(buf) - 1);
  if (copied == 0) {
    return false;
  }
  buf[std::min(copied, sizeof(buf) - 1)] = 0;
  *out = buf;
  return true;
}

void SyscallEntry(THREADID tid, CONTEXT *ctx, SYSCALL_STANDARD std, VOID *) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || g_taint_file.empty()) {
    return;
  }
  td->last_sys_num = PIN_GetSyscallNumber(ctx, std);
  td->last_sys_arg0 = PIN_GetSyscallArgument(ctx, std, 0);
  td->last_sys_arg1 = PIN_GetSyscallArgument(ctx, std, 1);
  td->last_sys_arg2 = PIN_GetSyscallArgument(ctx, std, 2);
  td->last_sys_arg4 = PIN_GetSyscallArgument(ctx, std, 4);
  td->last_open_match = false;

  if (td->last_sys_num == static_cast<ADDRINT>(__NR_open)) {
    std::string path;
    td->last_open_match =
        ReadUserString(td->last_sys_arg0, &path) && pintrace::PathMatches(path, g_taint_file);
  } else if (td->last_sys_num == static_cast<ADDRINT>(__NR_openat)) {
    std::string path;
    td->last_open_match =
        ReadUserString(td->last_sys_arg1, &path) && pintrace::PathMatches(path, g_taint_file);
  }
}

void SyscallExit(THREADID tid, CONTEXT *ctx, SYSCALL_STANDARD std, VOID *) {
  auto *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  if (td == nullptr || g_taint_file.empty()) {
    return;
  }
  const ADDRINT ret = PIN_GetSyscallReturn(ctx, std);
  const ADDRINT num = td->last_sys_num;
  if ((num == static_cast<ADDRINT>(__NR_open) || num == static_cast<ADDRINT>(__NR_openat)) &&
      td->last_open_match && static_cast<long>(ret) >= 0) {
    g_taint_fds.insert(static_cast<int>(ret));
    return;
  }
  if (num == static_cast<ADDRINT>(__NR_close)) {
    g_taint_fds.erase(static_cast<int>(td->last_sys_arg0));
    return;
  }
  if ((num == static_cast<ADDRINT>(__NR_read) || num == static_cast<ADDRINT>(__NR_pread64)) &&
      static_cast<long>(ret) > 0 && g_taint_fds.count(static_cast<int>(td->last_sys_arg0))) {
    MemSetTaint(static_cast<uint64_t>(td->last_sys_arg1), static_cast<uint32_t>(ret), true);
    return;
  }
  if (num == static_cast<ADDRINT>(__NR_mmap) && static_cast<long>(ret) >= 0 &&
      g_taint_fds.count(static_cast<int>(td->last_sys_arg4))) {
    MemSetTaint(static_cast<uint64_t>(ret), static_cast<uint32_t>(td->last_sys_arg1), true);
  }
}

void Instruction(INS ins, VOID *) {
  PIN_GetLock(&g_ipmap_lock, 1);
  pintrace::RememberInstruction(&g_ipmap, ins);
  PIN_ReleaseLock(&g_ipmap_lock);

  if (INS_HasScatteredMemoryAccess(ins) || INS_IsVscatter(ins) || INS_IsVgather(ins)) {
    return;
  }
  const UINT32 mem_ops = INS_MemoryOperandCount(ins);
  bool has_read = false;
  bool has_write = false;
  for (UINT32 mem_op = 0; mem_op < mem_ops; ++mem_op) {
    has_read = has_read || INS_MemoryOperandIsRead(ins, mem_op);
    has_write = has_write || INS_MemoryOperandIsWritten(ins, mem_op);
  }
  if (!has_write && INS_MaxNumWRegs(ins) == 0) {
    return;
  }

  INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintBegin), IARG_THREAD_ID, IARG_END);

  std::set<REG> address_regs;
  const REG base = NormReg(INS_MemoryBaseReg(ins));
  const REG index = NormReg(INS_MemoryIndexReg(ins));
  if (base != REG_INVALID()) {
    address_regs.insert(base);
  }
  if (index != REG_INVALID()) {
    address_regs.insert(index);
  }

  for (UINT32 i = 0; i < INS_MaxNumRRegs(ins); ++i) {
    const REG reg = NormReg(INS_RegR(ins, i));
    if (reg != REG_INVALID() && address_regs.count(reg) == 0) {
      INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccReg), IARG_THREAD_ID,
                     IARG_UINT32, static_cast<UINT32>(reg), IARG_END);
    }
  }
  if (has_read) {
    for (UINT32 mem_op = 0; mem_op < mem_ops; ++mem_op) {
      if (!INS_MemoryOperandIsRead(ins, mem_op)) {
        continue;
      }
      const UINT32 size = static_cast<UINT32>(INS_MemoryOperandSize(ins, mem_op));
      if (size > 0) {
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccMem),
                                 IARG_THREAD_ID, IARG_MEMORYOP_EA, mem_op,
                                 IARG_UINT32, size, IARG_END);
      }
    }
  }

  if (INS_IsValidForIpointAfter(ins)) {
    for (UINT32 mem_op = 0; mem_op < mem_ops && mem_op < ThreadData::kMaxMemOps; ++mem_op) {
      if (!INS_MemoryOperandIsWritten(ins, mem_op)) {
        continue;
      }
      const UINT32 size = static_cast<UINT32>(INS_MemoryOperandSize(ins, mem_op));
      if (size == 0) {
        continue;
      }
      INS_InsertPredicatedCall(ins, IPOINT_BEFORE, AFUNPTR(SaveWriteBefore),
                               IARG_THREAD_ID, IARG_UINT32, mem_op,
                               IARG_MEMORYOP_EA, mem_op, IARG_UINT32, size,
                               IARG_INST_PTR, IARG_END);
      INS_InsertPredicatedCall(ins, IPOINT_AFTER, AFUNPTR(CompareWriteAfter),
                               IARG_THREAD_ID, IARG_UINT32, mem_op,
                               IARG_INST_PTR, IARG_END);
    }
  }

  for (UINT32 i = 0; i < INS_MaxNumWRegs(ins); ++i) {
    const REG reg = NormReg(INS_RegW(ins, i));
    if (reg != REG_INVALID()) {
      INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetReg), IARG_THREAD_ID,
                     IARG_UINT32, static_cast<UINT32>(reg), IARG_END);
    }
  }
}

pintrace::SiteMeta MetaFor(uint64_t ip) {
  auto it = g_ipmap.find(ip);
  if (it != g_ipmap.end()) {
    return it->second;
  }
  pintrace::SiteMeta meta;
  meta.ip = ip;
  meta.ip_hex = pintrace::Hex(ip);
  return meta;
}

void WriteOwner(std::ofstream &out, const pintrace::SiteMeta &meta, uint64_t count,
                const std::string &indent) {
  out << indent << "{\n";
  out << indent << "  \"ip\": " << meta.ip << ",\n";
  out << indent << "  \"ip_hex\": \"" << pintrace::JsonEscape(meta.ip_hex) << "\",\n";
  out << indent << "  \"module\": \"" << pintrace::JsonEscape(meta.image) << "\",\n";
  out << indent << "  \"offset\": \"" << pintrace::JsonEscape(meta.offset) << "\",\n";
  out << indent << "  \"symbol\": \"" << pintrace::JsonEscape(meta.symbol) << "\",\n";
  out << indent << "  \"routine\": \"" << pintrace::JsonEscape(meta.routine) << "\",\n";
  out << indent << "  \"disasm\": \"" << pintrace::JsonEscape(meta.disasm) << "\",\n";
  out << indent << "  \"write_count\": " << count << "\n";
  out << indent << "}";
}

void WriteJson() {
  std::ofstream out(KnobOutput.Value().c_str());
  out << "{\n";
  out << "  \"taint_file\": \"" << pintrace::JsonEscape(g_taint_file) << "\",\n";
  out << "  \"taint_seed_mode\": \"file\",\n";
  out << "  \"block_size\": " << pintrace::kBlockSize << ",\n";

  if (g_site_bits_only) {
    out << "  \"site_bits_only\": true,\n";
    out << "  \"site_bits_txt\": \"" << pintrace::JsonEscape(KnobSiteBitsText.Value()) << "\",\n";
    out << "  \"address_count\": 0,\n";
    out << "  \"site_bit_definition\": \"Per-exact-write-site 01 string over taint-filtered block-overlap writes. 0 means unchanged, 1 means changed.\",\n";
    out << "  \"addresses\": [],\n";
    out << "  \"site_count\": " << g_sites.size() << ",\n";
    out << "  \"sites\": [\n";
    bool first = true;
    std::string first_bits;
    for (const auto &item : g_sites) {
      const SiteState &state = item.second;
      const pintrace::SiteMeta meta = MetaFor(item.first);
      if (!first) {
        out << ",\n";
      }
      first = false;
      if (first_bits.empty()) {
        first_bits = state.bits;
      }
      const double ratio = state.write_count == 0
                               ? 0.0
                               : static_cast<double>(state.unchanged_count) /
                                     static_cast<double>(state.write_count);
      out << "    {\n";
      out << "      \"ip\": " << meta.ip << ",\n";
      out << "      \"ip_hex\": \"" << pintrace::JsonEscape(meta.ip_hex) << "\",\n";
      out << "      \"module\": \"" << pintrace::JsonEscape(meta.image) << "\",\n";
      out << "      \"offset\": \"" << pintrace::JsonEscape(meta.offset) << "\",\n";
      out << "      \"symbol\": \"" << pintrace::JsonEscape(meta.symbol) << "\",\n";
      out << "      \"routine\": \"" << pintrace::JsonEscape(meta.routine) << "\",\n";
      out << "      \"disasm\": \"" << pintrace::JsonEscape(meta.disasm) << "\",\n";
      out << "      \"compare_count\": " << state.write_count << ",\n";
      out << "      \"bits_len\": " << state.bits.size() << ",\n";
      out << "      \"unchanged_count\": " << state.unchanged_count << ",\n";
      out << "      \"changed_count\": " << state.changed_count << ",\n";
      out << "      \"unchanged_ratio\": " << ratio << ",\n";
      out << "      \"bits\": \"" << state.bits << "\"\n";
      out << "    }";
    }
    out << "\n  ]\n";
    out << "}\n";
    if (!KnobSiteBitsText.Value().empty()) {
      std::ofstream bits(KnobSiteBitsText.Value().c_str());
      bits << first_bits;
    }
    return;
  }

  out << "  \"address_count\": " << g_blocks.size() << ",\n";
  out << "  \"bit_definition\": \"Per-tainted-16B-address 01 string over writes. 0 means unchanged, 1 means changed.\",\n";
  out << "  \"addresses\": [\n";
  bool first_block = true;
  for (const auto &item : g_blocks) {
    const BlockState &state = item.second;
    if (!first_block) {
      out << ",\n";
    }
    first_block = false;
    const double ratio = state.write_count == 0
                             ? 0.0
                             : static_cast<double>(state.unchanged_count) /
                                   static_cast<double>(state.write_count);
    auto owner_it = state.owner_counts.empty() ? state.owner_counts.end()
                                               : std::max_element(
                                                     state.owner_counts.begin(),
                                                     state.owner_counts.end(),
                                                     [](const auto &a, const auto &b) {
                                                       return a.second < b.second;
                                                     });
    out << "    {\n";
    out << "      \"paddr\": " << state.paddr << ",\n";
    out << "      \"paddr_hex\": \"" << pintrace::Hex(state.paddr) << "\",\n";
    out << "      \"size\": " << pintrace::kBlockSize << ",\n";
    out << "      \"write_count\": " << state.write_count << ",\n";
    out << "      \"compare_count\": " << state.write_count << ",\n";
    out << "      \"bits_len\": " << state.bits.size() << ",\n";
    out << "      \"has_initial_baseline\": true,\n";
    out << "      \"unchanged_count\": " << state.unchanged_count << ",\n";
    out << "      \"changed_count\": " << state.changed_count << ",\n";
    out << "      \"unchanged_ratio\": " << ratio << ",\n";
    out << "      \"bits\": \"" << state.bits << "\",\n";
    if (owner_it != state.owner_counts.end()) {
      out << "      \"owner\": ";
      WriteOwner(out, MetaFor(owner_it->first), owner_it->second, "      ");
      out << ",\n";
    } else {
      out << "      \"owner\": null,\n";
    }
    out << "      \"owner_candidate_count\": " << state.owner_counts.size() << ",\n";
    out << "      \"owner_candidates\": [\n";
    bool first_owner = true;
    for (const auto &owner : state.owner_counts) {
      if (!first_owner) {
        out << ",\n";
      }
      first_owner = false;
      WriteOwner(out, MetaFor(owner.first), owner.second, "        ");
    }
    out << "\n      ]\n";
    out << "    }";
  }
  out << "\n  ]\n";
  out << "}\n";
}

void ThreadStart(THREADID tid, CONTEXT *, INT32, VOID *) {
  auto *td = new ThreadData();
  td->reg_taint.resize(static_cast<size_t>(REG_LAST));
  PIN_SetThreadData(g_tls_key, td, tid);
}

void ThreadFini(THREADID tid, const CONTEXT *, INT32, VOID *) {
  delete static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
  PIN_SetThreadData(g_tls_key, nullptr, tid);
}

void Fini(INT32, VOID *) {
  WriteJson();
  pintrace::WriteIpMap(KnobIpMap.Value(), g_ipmap);
  if (g_pagemap_fd >= 0) {
    close(g_pagemap_fd);
    g_pagemap_fd = -1;
  }
  for (auto &item : g_shadow_pages) {
    delete item.second;
  }
  g_shadow_pages.clear();
}

INT32 Usage() {
  std::fprintf(stderr,
               "taintblock16trace: collect taint-filtered 16B block 01 strings.\n"
               "  -o <file>                 JSON output\n"
	               "  -m <file>                 optional IP map\n"
	               "  -taint-file <file>        file/basename used to seed taint\n"
	               "  -site-ip <hex>            optional exact write-site filter\n"
		               "  -site-bits-only 0|1       emit only selected site string\n"
		               "  -site-bits-txt <file>     raw 01 site string output\n");
  return -1;
}

}  // namespace

int main(int argc, char *argv[]) {
  PIN_InitSymbols();
  if (PIN_Init(argc, argv)) {
    return Usage();
  }
  if (KnobTaintSeedMode.Value() != "file") {
    std::fprintf(stderr, "artifact taintblock16trace keeps only -taint-seed-mode file\n");
    return 1;
  }
  g_taint_file = KnobTaintFile.Value();
  g_site_bits_only = KnobSiteBitsOnly.Value();
  if (!KnobSiteIp.Value().empty()) {
    if (!pintrace::ParseU64(KnobSiteIp.Value(), &g_site_filter)) {
      std::fprintf(stderr, "invalid -site-ip: %s\n", KnobSiteIp.Value().c_str());
      return 1;
    }
    g_has_site_filter = true;
  }
  if (g_site_bits_only && !g_has_site_filter) {
    std::fprintf(stderr, "-site-bits-only requires -site-ip\n");
    return 1;
  }
  if (KnobStackDepth.Value() != 0) {
    std::fprintf(stderr, "warning: stack capture is omitted in artifact build\n");
  }
  g_page_size = static_cast<uint64_t>(::getpagesize());
  g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
  if (g_pagemap_fd < 0) {
    std::fprintf(stderr, "failed to open /proc/self/pagemap; physical address mode needs sufficient privileges\n");
    return 1;
  }

  PIN_InitLock(&g_stream_lock);
  PIN_InitLock(&g_taint_lock);
  PIN_InitLock(&g_ipmap_lock);
  g_tls_key = PIN_CreateThreadDataKey(nullptr);
  INS_AddInstrumentFunction(Instruction, nullptr);
  PIN_AddSyscallEntryFunction(SyscallEntry, nullptr);
  PIN_AddSyscallExitFunction(SyscallExit, nullptr);
  PIN_AddThreadStartFunction(ThreadStart, nullptr);
  PIN_AddThreadFiniFunction(ThreadFini, nullptr);
  PIN_AddFiniFunction(Fini, nullptr);
  PIN_StartProgram();
  return 0;
}
