#include "pin.H"

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <elf.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

static KNOB<std::string> KnobOutput(KNOB_MODE_WRITEONCE, "pintool", "o",
                                    "paddrtrace.bin", "Output trace file");
static KNOB<std::string> KnobIpMap(KNOB_MODE_WRITEONCE, "pintool", "m",
                                   "paddrtrace.ip.txt",
                                   "Instruction IP map output");
static KNOB<UINT32> KnobStackDepth(KNOB_MODE_WRITEONCE, "pintool", "stack-depth",
                                   "16", "Call stack depth per record (0 disables)");
static KNOB<std::string> KnobSiteIp(
    KNOB_MODE_WRITEONCE, "pintool", "site-ip", "",
    "Optional exact write-site IP filter. Only matching writes are emitted.");
static KNOB<std::string> KnobStackContainsIp(
    KNOB_MODE_WRITEONCE, "pintool", "stack-contains-ip", "",
    "Optional callsite IP filter. Only writes whose current callstack "
    "contains this callsite anywhere are emitted.");
static KNOB<BOOL> KnobAfterOnly(
    KNOB_MODE_WRITEONCE, "pintool", "after-only", "0",
    "Only emit post-write records. Skips init and before-only records.");

static KNOB<BOOL> KnobNoPaddr(
    KNOB_MODE_WRITEONCE, "pintool", "no-paddr", "0",
    "Disable /proc/self/pagemap (no paddr/paddr16). Avoids needing sudo.");

static KNOB<std::string> KnobTaintFile(
    KNOB_MODE_WRITEONCE, "pintool", "taint-file", "",
    "Seed taint from bytes read() from this file (path or basename). Empty disables taint.");
static KNOB<std::string> KnobTaintSeedMode(
    KNOB_MODE_WRITEONCE, "pintool", "taint-seed-mode", "file",
    "Taint seed mode: file|input-tensor. file=seed from read() of -taint-file. input-tensor=seed from Glow input placeholder copy.");
static KNOB<BOOL> KnobTaintOnly(
    KNOB_MODE_WRITEONCE, "pintool", "taint-only", "0",
    "Only emit records for stores whose written value is tainted (input-derived).");
static KNOB<std::string> KnobTaintDecimalOut(
    KNOB_MODE_WRITEONCE, "pintool", "taint-decimal-out", "",
    "Optional: write a text log of tainted stores with values shown in decimal.");
static KNOB<std::string> KnobTaintDecimalFmt(
    KNOB_MODE_WRITEONCE, "pintool", "taint-decimal-fmt", "auto",
    "Main text format for tainted values: auto|u8|u16|u32|u64|hex|f32|f64. "
    "auto uses a conservative integer/raw view and emits f32 as an auxiliary view.");
static KNOB<BOOL> KnobTaintNoLock(
    KNOB_MODE_WRITEONCE, "pintool", "taint-no-lock", "0",
    "Disable locks for taint shadow state (unsafe if multiple threads, faster).");

static PIN_LOCK g_lock;
static PIN_LOCK g_seen_lock;
static PIN_LOCK g_inst_lock;
static PIN_LOCK g_taint_lock;
static FILE *g_out = nullptr;
static FILE *g_taint_txt = nullptr;
static int g_pagemap_fd = -1;
static size_t g_page_size = 0;
static std::atomic<uint64_t> g_seq{0};
static TLS_KEY g_tls_key = INVALID_TLS_KEY;
static std::unordered_set<uint64_t> g_seen_paddr16;
// Record pre-write bytes (BEFORE records) only once per tainted (vaddr,size,size_read)
// key, to capture the pre-write value for the first tainted write to that key.
struct FirstTaintKey {
    uint64_t vaddr;
    uint32_t size;
    uint32_t size_read;
};
struct FirstTaintKeyEq {
    bool operator()(const FirstTaintKey &a, const FirstTaintKey &b) const {
        return a.vaddr == b.vaddr && a.size == b.size && a.size_read == b.size_read;
    }
};
struct FirstTaintKeyHash {
    size_t operator()(const FirstTaintKey &k) const {
        // Simple mix: vaddr dominates; include size/size_read.
        const uint64_t x = k.vaddr ^ (static_cast<uint64_t>(k.size) << 32) ^
                           static_cast<uint64_t>(k.size_read);
        // xorshift64*
        uint64_t z = x + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return static_cast<size_t>(z);
    }
};
static std::unordered_set<FirstTaintKey, FirstTaintKeyHash, FirstTaintKeyEq>
    g_seen_first_taint_before;
static std::map<uint64_t, std::string> g_inst_map;
static std::atomic<uint64_t> g_current_instr_id{UINT64_MAX};
static std::atomic<ADDRINT> g_instr_id_addr{0};
static bool g_no_paddr = false;
static bool g_taint_enabled = false;
static bool g_taint_only = false;
static bool g_taint_no_lock = false;
static bool g_after_only = false;
static bool g_has_site_ip_filter = false;
static uint64_t g_site_ip_filter = 0;
static bool g_has_stack_contains_ip_filter = false;
static uint64_t g_stack_contains_ip_filter = 0;
static std::string g_taint_file;
static std::string g_taint_file_base;
static std::unordered_set<int> g_taint_fds;
static std::string g_taint_seed_mode;
static bool g_seed_from_input_tensor = false;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];      // "PADDRTRC"
    uint32_t version;   // 2
    uint32_t reserved;  // 0
    uint64_t page_size;
};

struct RecordHeader {
    uint64_t seq;
    uint32_t tid;
    uint32_t size;         // bytes written by the instruction
    uint32_t size_read;    // bytes successfully read from write address
    uint32_t block16_read; // bytes successfully read from 16B aligned block
    uint64_t vaddr;
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

struct ThreadData {
    std::vector<uint8_t> write_buf;
    std::array<uint8_t, 16> block16;
    std::unordered_map<uint64_t, uint64_t> pfn_cache;
    std::vector<uint64_t> callstack;
    std::vector<uint64_t> stack_ips;

    // Some instructions (e.g., push/call/rep stos*) both write memory and
    // modify the base/index register used to compute the memory operand EA.
    // If we query IARG_MEMORYOP_EA at IPOINT_AFTER, the returned EA may be
    // computed using post-instruction register values, yielding a consistent
    // offset error (e.g., -8 bytes for push rbx). To preserve "write-after"
    // value semantics while keeping the correct address, we snapshot the EA
    // at IPOINT_BEFORE and then read the written bytes at IPOINT_AFTER using
    // the saved EA.
    static constexpr size_t kMaxSavedMemOps = 8;
    std::array<ADDRINT, kMaxSavedMemOps> saved_write_ea{};
    std::array<UINT32, kMaxSavedMemOps> saved_write_size{};
    std::array<uint8_t, kMaxSavedMemOps> saved_write_valid{};

    // --- Taint tracking state (1-bit taint; conservative) ---
    bool cur_taint = false;
    std::vector<uint8_t> reg_taint;

    uint32_t update_inputs_depth = 0;
    uint32_t tensor_assign_depth = 0;

    // Syscall tracking (for seeding taint from read()/mmap() of the input file).
    ADDRINT last_sys_num = 0;
    ADDRINT last_sys_arg0 = 0;
    ADDRINT last_sys_arg1 = 0;
    ADDRINT last_sys_arg2 = 0;
    ADDRINT last_sys_arg3 = 0;
    ADDRINT last_sys_arg4 = 0;
    bool last_open_match = false;
};

static uint64_t ReadInstrId();
static bool FindSymbolInElf(const std::string &path, const char *sym_name,
                            uint64_t *value_out, bool *is_pie_out);
static VOID RecordWrite(THREADID tid, ADDRINT addr, UINT32 size, BOOL is_after,
                        ADDRINT ip);
static VOID RecordFirstTaintBefore(THREADID tid, ADDRINT addr, UINT32 size,
                                   ADDRINT ip);

static VOID SaveWriteEA(THREADID tid, UINT32 mem_op, ADDRINT ea, UINT32 size) {
    ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (mem_op >= td->saved_write_ea.size()) {
        return;
    }
    td->saved_write_ea[mem_op] = ea;
    td->saved_write_size[mem_op] = size;
    td->saved_write_valid[mem_op] = 1;
}

static VOID RecordWriteSaved(THREADID tid, UINT32 mem_op, BOOL is_after,
                             ADDRINT ip) {
    ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (mem_op >= td->saved_write_ea.size()) {
        return;
    }
    if (!td->saved_write_valid[mem_op]) {
        return;
    }
    const ADDRINT ea = td->saved_write_ea[mem_op];
    const UINT32 size = td->saved_write_size[mem_op];
    td->saved_write_valid[mem_op] = 0;
    RecordWrite(tid, ea, size, is_after, ip);
}

static INT32 Usage() {
    std::fprintf(stderr,
                 "paddrtrace: records each memory write with physical address\n"
                 "  -o <file>  output trace file (default: paddrtrace.bin)\n"
                 "  -m <file>  IP map output (default: paddrtrace.ip.txt)\n"
                 "  -stack-depth <N>  call stack depth per record (0 disables)\n"
                 "  -site-ip <addr>    only emit writes from this exact site IP\n"
                 "  -stack-contains-ip <addr>  require this callsite in current callstack\n"
                 "  -after-only 0|1    only emit post-write records\n"
                 "  -no-paddr 0|1      disable /proc/self/pagemap (no paddr/paddr16; no sudo)\n"
                 "  -taint-file <path>  seed taint from read() of this file (optional)\n"
                 "  -taint-seed-mode file|input-tensor  taint seed mode\n"
                 "  -taint-only 0|1     only emit tainted stores (optional)\n"
                 "  -taint-decimal-out <file>  write tainted values as decimals (optional)\n"
                 "  -taint-decimal-fmt auto|u8|u32|f32|f64  decimal encoding\n");
    return -1;
}

static std::string BaseName(const std::string &path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

static bool ParseUint64Arg(const std::string &text, uint64_t *out) {
    if (out == nullptr || text.empty()) {
        return false;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 0);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    *out = static_cast<uint64_t>(parsed);
    return true;
}

static bool PathMatchesTaintFile(const std::string &path) {
    if (!g_taint_enabled) {
        return false;
    }
    if (path == g_taint_file) {
        return true;
    }
    const std::string base = BaseName(path);
    return (!g_taint_file_base.empty() && base == g_taint_file_base);
}

struct ShadowPage {
    // 4096 bytes -> 4096 bits -> 64 * uint64_t.
    std::array<uint64_t, 64> bits{};
};

static std::unordered_map<uint64_t, ShadowPage *> g_shadow_pages;

static ShadowPage *GetShadowPage(uint64_t page_no, bool create) {
    auto it = g_shadow_pages.find(page_no);
    if (it != g_shadow_pages.end()) {
        return it->second;
    }
    if (!create) {
        return nullptr;
    }
    ShadowPage *p = new ShadowPage();
    g_shadow_pages.emplace(page_no, p);
    return p;
}

static inline void ShadowSetByte(ShadowPage *p, uint32_t byte_off,
                                 bool tainted) {
    const uint32_t bit = byte_off & 63u;
    const uint32_t word = byte_off >> 6;
    const uint64_t mask = (1ULL << bit);
    if (tainted) {
        p->bits[word] |= mask;
    } else {
        p->bits[word] &= ~mask;
    }
}

static inline bool ShadowGetByte(const ShadowPage *p, uint32_t byte_off) {
    const uint32_t bit = byte_off & 63u;
    const uint32_t word = byte_off >> 6;
    return ((p->bits[word] >> bit) & 1ULL) != 0;
}

static bool MemAnyTaint(uint64_t addr, uint32_t size) {
    if (size == 0) {
        return false;
    }
    const uint64_t end = addr + static_cast<uint64_t>(size);
    for (uint64_t cur = addr; cur < end; ++cur) {
        const uint64_t page = cur >> 12;
        const uint32_t off = static_cast<uint32_t>(cur & 0xFFF);
        const ShadowPage *p = nullptr;
        if (!g_taint_no_lock) {
            PIN_GetLock(&g_taint_lock, 1);
        }
        auto it = g_shadow_pages.find(page);
        if (it != g_shadow_pages.end()) {
            p = it->second;
        }
        if (!g_taint_no_lock) {
            PIN_ReleaseLock(&g_taint_lock);
        }
        if (p != nullptr && ShadowGetByte(p, off)) {
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
        const uint64_t page = cur >> 12;
        const uint32_t off = static_cast<uint32_t>(cur & 0xFFF);
        if (!g_taint_no_lock) {
            PIN_GetLock(&g_taint_lock, 1);
        }
        ShadowPage *p = GetShadowPage(page, tainted /*create*/);
        if (p != nullptr) {
            ShadowSetByte(p, off, tainted);
        }
        if (!g_taint_no_lock) {
            PIN_ReleaseLock(&g_taint_lock);
        }
    }
}

static void MemSetTaintRange(uint64_t addr, uint64_t size, bool tainted) {
    while (size > 0) {
        const uint32_t chunk =
            static_cast<uint32_t>(std::min<uint64_t>(size, UINT32_MAX));
        MemSetTaint(addr, chunk, tainted);
        addr += static_cast<uint64_t>(chunk);
        size -= static_cast<uint64_t>(chunk);
    }
}

static inline REG NormReg(REG r) {
    if (r == REG_INVALID()) {
        return r;
    }
    return REG_FullRegName(r);
}

static inline bool GetRegTaint(ThreadData *td, REG r) {
    r = NormReg(r);
    if (r == REG_INVALID()) {
        return false;
    }
    const uint32_t idx = static_cast<uint32_t>(r);
    if (idx >= td->reg_taint.size()) {
        return false;
    }
    return td->reg_taint[idx] != 0;
}

static inline void SetRegTaint(ThreadData *td, REG r, bool tainted) {
    r = NormReg(r);
    if (r == REG_INVALID()) {
        return;
    }
    const uint32_t idx = static_cast<uint32_t>(r);
    if (idx >= td->reg_taint.size()) {
        return;
    }
    td->reg_taint[idx] = tainted ? 1 : 0;
}

static VOID TaintBegin(THREADID tid) {
    if (!g_taint_enabled) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    td->cur_taint = false;
}

static VOID TaintAccReg(THREADID tid, UINT32 reg_u32) {
    if (!g_taint_enabled) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    const REG r = static_cast<REG>(reg_u32);
    if (GetRegTaint(td, r)) {
        td->cur_taint = true;
    }
}

static VOID TaintAccMem(THREADID tid, ADDRINT addr, UINT32 size) {
    if (!g_taint_enabled) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (MemAnyTaint(static_cast<uint64_t>(addr), size)) {
        td->cur_taint = true;
    }
}

static VOID TaintSetReg(THREADID tid, UINT32 reg_u32) {
    if (!g_taint_enabled) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    const REG r = static_cast<REG>(reg_u32);
    SetRegTaint(td, r, td->cur_taint);
}

static VOID TaintSetMem(THREADID tid, ADDRINT addr, UINT32 size) {
    if (!g_taint_enabled) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    MemSetTaint(static_cast<uint64_t>(addr), size, td->cur_taint);
}

static VOID OnEnterUpdateInputs(THREADID tid) {
    if (!g_seed_from_input_tensor) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    td->update_inputs_depth++;
}

static VOID OnExitUpdateInputs(THREADID tid) {
    if (!g_seed_from_input_tensor) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (td->update_inputs_depth > 0) {
        td->update_inputs_depth--;
    }
}

static VOID OnEnterTensorAssign(THREADID tid) {
    if (!g_seed_from_input_tensor) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (td->update_inputs_depth == 0) {
        return;
    }
    td->tensor_assign_depth++;
}

static VOID OnExitTensorAssign(THREADID tid) {
    if (!g_seed_from_input_tensor) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (td->tensor_assign_depth > 0) {
        td->tensor_assign_depth--;
    }
}

static VOID SyscallEntry(THREADID tid, CONTEXT *ctx, SYSCALL_STANDARD std,
                         VOID *) {
    if (!g_taint_enabled) {
        return;
    }
    if (g_seed_from_input_tensor) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
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
        const char *path_ptr =
            reinterpret_cast<const char *>(td->last_sys_arg0);
        char tmp[512];
        tmp[0] = 0;
        PIN_SafeCopy(tmp, path_ptr, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        td->last_open_match = PathMatchesTaintFile(std::string(tmp));
    } else if (td->last_sys_num == static_cast<ADDRINT>(__NR_openat)) {
        const char *path_ptr =
            reinterpret_cast<const char *>(td->last_sys_arg1);
        char tmp[512];
        tmp[0] = 0;
        PIN_SafeCopy(tmp, path_ptr, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        td->last_open_match = PathMatchesTaintFile(std::string(tmp));
    }
}

static VOID SyscallExit(THREADID tid, CONTEXT *ctx, SYSCALL_STANDARD std,
                        VOID *) {
    if (!g_taint_enabled) {
        return;
    }
    if (g_seed_from_input_tensor) {
        return;
    }
    ThreadData *td =
        static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }

    const ADDRINT num = td->last_sys_num;
    const ADDRINT ret = PIN_GetSyscallReturn(ctx, std);
    auto is_taint_fd = [&](int fd) {
        bool tagged = false;
        if (!g_taint_no_lock) {
            PIN_GetLock(&g_taint_lock, 1);
        }
        tagged = (g_taint_fds.find(fd) != g_taint_fds.end());
        if (!g_taint_no_lock) {
            PIN_ReleaseLock(&g_taint_lock);
        }
        return tagged;
    };
    auto set_taint_fd = [&](int fd, bool tagged) {
        if (!g_taint_no_lock) {
            PIN_GetLock(&g_taint_lock, 1);
        }
        if (tagged) {
            g_taint_fds.insert(fd);
        } else {
            g_taint_fds.erase(fd);
        }
        if (!g_taint_no_lock) {
            PIN_ReleaseLock(&g_taint_lock);
        }
    };

    if (num == static_cast<ADDRINT>(__NR_open) ||
        num == static_cast<ADDRINT>(__NR_openat)) {
        if (td->last_open_match && static_cast<long>(ret) >= 0) {
            const int fd = static_cast<int>(ret);
            set_taint_fd(fd, true);
        }
        return;
    }

    if (num == static_cast<ADDRINT>(__NR_close)) {
        const int fd = static_cast<int>(td->last_sys_arg0);
        set_taint_fd(fd, false);
        return;
    }

    if (num == static_cast<ADDRINT>(__NR_mmap)) {
        const int fd = static_cast<int>(td->last_sys_arg4);
        const uint64_t length = static_cast<uint64_t>(td->last_sys_arg1);
        if (fd >= 0 && length > 0 && static_cast<long>(ret) >= 0 &&
            is_taint_fd(fd)) {
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

    if (static_cast<long>(ret) >= 0 && num == static_cast<ADDRINT>(__NR_dup)) {
            const int oldfd = static_cast<int>(td->last_sys_arg0);
            const int newfd = static_cast<int>(ret);
            if (oldfd != newfd) {
                set_taint_fd(newfd, is_taint_fd(oldfd));
            }
            return;
    }
    if (static_cast<long>(ret) >= 0 &&
        (num == static_cast<ADDRINT>(__NR_dup2) ||
         num == static_cast<ADDRINT>(__NR_dup3))) {
        const int oldfd = static_cast<int>(td->last_sys_arg0);
        const int newfd = static_cast<int>(td->last_sys_arg1);
        set_taint_fd(newfd, is_taint_fd(oldfd));
        return;
    }
    if (static_cast<long>(ret) >= 0 && num == static_cast<ADDRINT>(__NR_fcntl)) {
        const int fd = static_cast<int>(td->last_sys_arg0);
        const int cmd = static_cast<int>(td->last_sys_arg1);
        if (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC) {
            const int newfd = static_cast<int>(ret);
            if (fd != newfd) {
                set_taint_fd(newfd, is_taint_fd(fd));
            }
            return;
        }
    }

    if (num == static_cast<ADDRINT>(__NR_read) ||
        num == static_cast<ADDRINT>(__NR_pread64)) {
        const int fd = static_cast<int>(td->last_sys_arg0);
        const uint64_t buf = static_cast<uint64_t>(td->last_sys_arg1);
        const long nread = static_cast<long>(ret);

        if (is_taint_fd(fd) && nread > 0 &&
            nread <= static_cast<long>(std::numeric_limits<uint32_t>::max())) {
            MemSetTaint(buf, static_cast<uint32_t>(nread), true);
        }
    }
}

static bool ReadPagemapEntry(uint64_t vpn, uint64_t *entry_out) {
    const off_t offset = static_cast<off_t>(vpn * 8);
    uint64_t entry = 0;
    const ssize_t n = pread(g_pagemap_fd, &entry, sizeof(entry), offset);
    if (n != static_cast<ssize_t>(sizeof(entry))) {
        return false;
    }
    *entry_out = entry;
    return true;
}

static uint64_t VaddrToPaddr(uint64_t vaddr, ThreadData *td, bool *ok) {
    if (g_pagemap_fd < 0) {
        *ok = false;
        return 0;
    }
    const uint64_t vpn = vaddr / g_page_size;
    auto it = td->pfn_cache.find(vpn);
    uint64_t entry = 0;
    if (it != td->pfn_cache.end()) {
        entry = it->second;
    } else {
        if (!ReadPagemapEntry(vpn, &entry)) {
            *ok = false;
            return 0;
        }
    }

    const bool present = (entry & (1ULL << 63)) != 0;
    if (!present) {
        if (it != td->pfn_cache.end()) {
            td->pfn_cache.erase(it);
        }
        *ok = false;
        return 0;
    }

    const uint64_t pfn = entry & ((1ULL << 55) - 1);
    if (pfn == 0) {
        if (it != td->pfn_cache.end()) {
            td->pfn_cache.erase(it);
        }
        *ok = false;
        return 0;
    }

    if (it == td->pfn_cache.end()) {
        td->pfn_cache.emplace(vpn, entry);
    }

    const uint64_t offset = vaddr % g_page_size;
    *ok = true;
    return (pfn * g_page_size) + offset;
}

static void EmitRecord(const RecordHeader &hdr, const uint8_t *write_data,
                       size_t write_len, const uint8_t *block16,
                       const uint64_t *stack_ips, uint32_t stack_depth) {
    PIN_GetLock(&g_lock, 1);
    std::fwrite(&hdr, sizeof(hdr), 1, g_out);
    if (write_len > 0) {
        std::fwrite(write_data, 1, write_len, g_out);
    }
    std::fwrite(block16, 1, 16, g_out);
    if (stack_depth > 0 && stack_ips != nullptr) {
        std::fwrite(stack_ips, sizeof(uint64_t), stack_depth, g_out);
    }
    PIN_ReleaseLock(&g_lock);
}

static uint32_t CaptureStack(ThreadData *td) {
    const uint32_t max_depth = KnobStackDepth.Value();
    if (max_depth == 0) {
        td->stack_ips.clear();
        return 0;
    }
    const size_t depth = td->callstack.size();
    if (depth == 0) {
        td->stack_ips.clear();
        return 0;
    }
    const size_t n = depth < max_depth ? depth : static_cast<size_t>(max_depth);
    td->stack_ips.resize(n);
    for (size_t i = 0; i < n; ++i) {
        td->stack_ips[i] = td->callstack[depth - 1 - i];
    }
    return static_cast<uint32_t>(n);
}

static bool CallstackContainsIp(const ThreadData *td, uint64_t ip) {
    if (td == nullptr) {
        return false;
    }
    for (size_t i = 0; i < td->callstack.size(); ++i) {
        if (td->callstack[i] == ip) {
            return true;
        }
    }
    return false;
}

static bool RecordPassesFilters(const ThreadData *td, uint64_t ip, bool is_after) {
    if (g_after_only && !is_after) {
        return false;
    }
    if (g_has_stack_contains_ip_filter &&
        !CallstackContainsIp(td, g_stack_contains_ip_filter)) {
        return false;
    }
    return true;
}

static VOID RecordInitIfFirst(THREADID tid, ADDRINT addr, UINT32 size,
                              ADDRINT ip) {
    if (g_after_only || g_has_site_ip_filter || g_has_stack_contains_ip_filter) {
        return;
    }
    if (g_taint_only) {
        return;
    }
    if (size == 0 || g_out == nullptr) {
        return;
    }

    ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }

    const uint64_t vaddr = static_cast<uint64_t>(addr);
    const uint64_t start = vaddr & ~0xFULL;
    const uint64_t last = (vaddr + static_cast<uint64_t>(size) - 1) & ~0xFULL;

    for (uint64_t cur = start; cur <= last; cur += 16) {
        bool ok_paddr16 = false;
        const uint64_t paddr16 = VaddrToPaddr(cur, td, &ok_paddr16);
        if (!ok_paddr16) {
            continue;
        }

        bool is_new = false;
        PIN_GetLock(&g_seen_lock, tid + 1);
        auto res = g_seen_paddr16.insert(paddr16);
        is_new = res.second;
        PIN_ReleaseLock(&g_seen_lock);

        if (!is_new) {
            continue;
        }

        const size_t block_read = PIN_SafeCopy(td->block16.data(),
                                               reinterpret_cast<const VOID *>(cur),
                                               td->block16.size());
        if (block_read < td->block16.size()) {
            std::memset(td->block16.data() + block_read, 0,
                        td->block16.size() - block_read);
        }

        RecordHeader hdr;
        hdr.seq = g_seq.fetch_add(1, std::memory_order_relaxed);
        hdr.tid = static_cast<uint32_t>(tid);
        hdr.size = 0;
        hdr.size_read = 0;
        hdr.block16_read = static_cast<uint32_t>(block_read);
        hdr.vaddr = cur;
        hdr.paddr = paddr16;
        hdr.paddr16 = paddr16;
        hdr.ip = static_cast<uint64_t>(ip);
        hdr.instr_id = ReadInstrId();
        hdr.flags = kFlagInitValue | kFlagPaddrValid | kFlagPaddr16Valid;
        if (block_read != td->block16.size()) {
            hdr.flags |= kFlagPartialBlock16Read;
        }
        const uint32_t stack_depth = CaptureStack(td);
        hdr.stack_depth = stack_depth;

        EmitRecord(hdr, nullptr, 0, td->block16.data(),
                   stack_depth ? td->stack_ips.data() : nullptr, stack_depth);
    }
}

static inline bool MarkFirstTaintBeforeSeen(THREADID tid, const FirstTaintKey &k) {
    bool is_new = false;
    PIN_GetLock(&g_seen_lock, tid + 1);
    auto res = g_seen_first_taint_before.insert(k);
    is_new = res.second;
    PIN_ReleaseLock(&g_seen_lock);
    return is_new;
}

static VOID RecordFirstTaintBefore(THREADID tid, ADDRINT addr, UINT32 size,
                                   ADDRINT ip) {
    // Record the pre-write payload bytes only for the FIRST tainted write to this
    // (vaddr,size,size_read) key. This captures the "pre" value without doubling
    // trace size for repeated writes.
    if (size == 0 || g_out == nullptr) {
        return;
    }

    ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (!RecordPassesFilters(td, static_cast<uint64_t>(ip), false)) {
        return;
    }

    const uint64_t vaddr = static_cast<uint64_t>(addr);
    const uint64_t start = vaddr & ~0xFULL;
    const uint64_t end = vaddr + static_cast<uint64_t>(size);
    const uint64_t last = (end - 1) & ~0xFULL;

    if (td->write_buf.size() < size) {
        td->write_buf.resize(size);
    }

    const size_t size_read = PIN_SafeCopy(td->write_buf.data(),
                                          reinterpret_cast<const VOID *>(vaddr),
                                          size);
    if (size_read < size) {
        std::memset(td->write_buf.data() + size_read, 0, size - size_read);
    }

    for (uint64_t cur = start; cur <= last; cur += 16) {
        const uint64_t block_start = cur;
        const uint64_t block_end = cur + 16;
        const uint64_t overlap_start = std::max(block_start, vaddr);
        const uint64_t overlap_end = std::min(block_end, end);
        if (overlap_end <= overlap_start) {
            continue;
        }

        const uint32_t overlap_len =
            static_cast<uint32_t>(overlap_end - overlap_start);
        const size_t offset = static_cast<size_t>(overlap_start - vaddr);

        // Snapshot pre-write 16B block (aligned).
        const size_t block_read = PIN_SafeCopy(
            td->block16.data(), reinterpret_cast<const VOID *>(cur),
            td->block16.size());
        if (block_read < td->block16.size()) {
            std::memset(td->block16.data() + block_read, 0,
                        td->block16.size() - block_read);
        }

        bool ok_paddr = false;
        bool ok_paddr16 = false;
        const uint64_t paddr = VaddrToPaddr(overlap_start, td, &ok_paddr);
        const uint64_t paddr16 = VaddrToPaddr(cur, td, &ok_paddr16);

        uint32_t part_read = 0;
        if (size_read > offset) {
            const size_t remaining = size_read - offset;
            part_read = static_cast<uint32_t>(
                std::min(static_cast<size_t>(overlap_len), remaining));
        }

        bool seeded_here = false;
        if (g_seed_from_input_tensor &&
            (td->tensor_assign_depth > 0 || td->update_inputs_depth > 0)) {
            // Same seeding behavior as RecordWrite().
            MemSetTaint(overlap_start, overlap_len, true);
            seeded_here = true;
        }

        const bool is_tainted =
            seeded_here ||
            (g_taint_enabled && MemAnyTaint(overlap_start, overlap_len));
        if (!is_tainted) {
            continue;
        }

        // Only emit the pre-write payload once per key.
        const FirstTaintKey key = {overlap_start, overlap_len, part_read};
        if (!MarkFirstTaintBeforeSeen(tid, key)) {
            continue;
        }

        RecordHeader hdr;
        hdr.seq = g_seq.fetch_add(1, std::memory_order_relaxed);
        hdr.tid = static_cast<uint32_t>(tid);
        hdr.size = overlap_len;
        hdr.size_read = part_read;
        hdr.block16_read = static_cast<uint32_t>(block_read);
        hdr.vaddr = overlap_start;
        hdr.paddr = ok_paddr ? paddr : 0;
        hdr.paddr16 = ok_paddr16 ? paddr16 : 0;
        hdr.ip = static_cast<uint64_t>(ip);
        hdr.instr_id = ReadInstrId();
        hdr.flags = kFlagUsedBefore | kFlagFirstTaintBefore;
        if (ok_paddr) {
            hdr.flags |= kFlagPaddrValid;
        }
        if (ok_paddr16) {
            hdr.flags |= kFlagPaddr16Valid;
        }
        if (part_read != overlap_len) {
            hdr.flags |= kFlagPartialWriteRead;
        }
        if (block_read != td->block16.size()) {
            hdr.flags |= kFlagPartialBlock16Read;
        }
        const uint32_t stack_depth = CaptureStack(td);
        hdr.stack_depth = stack_depth;

        EmitRecord(hdr, td->write_buf.data() + offset, overlap_len,
                   td->block16.data(),
                   stack_depth ? td->stack_ips.data() : nullptr, stack_depth);
    }
}

static VOID RecordWrite(THREADID tid, ADDRINT addr, UINT32 size, BOOL is_after,
                        ADDRINT ip) {
    if (size == 0 || g_out == nullptr) {
        return;
    }

    ThreadData *td = static_cast<ThreadData *>(PIN_GetThreadData(g_tls_key, tid));
    if (td == nullptr) {
        return;
    }
    if (!RecordPassesFilters(td, static_cast<uint64_t>(ip), is_after != 0)) {
        return;
    }

    const uint64_t vaddr = static_cast<uint64_t>(addr);
    const uint64_t start = vaddr & ~0xFULL;
    const uint64_t end = vaddr + static_cast<uint64_t>(size);
    const uint64_t last = (end - 1) & ~0xFULL;

    if (td->write_buf.size() < size) {
        td->write_buf.resize(size);
    }

    const size_t size_read = PIN_SafeCopy(td->write_buf.data(),
                                          reinterpret_cast<const VOID *>(vaddr),
                                          size);
    if (size_read < size) {
        std::memset(td->write_buf.data() + size_read, 0, size - size_read);
    }

    for (uint64_t cur = start; cur <= last; cur += 16) {
        const uint64_t block_start = cur;
        const uint64_t block_end = cur + 16;
        const uint64_t overlap_start = std::max(block_start, vaddr);
        const uint64_t overlap_end = std::min(block_end, end);
        if (overlap_end <= overlap_start) {
            continue;
        }

        const uint32_t overlap_len =
            static_cast<uint32_t>(overlap_end - overlap_start);
        const size_t offset = static_cast<size_t>(overlap_start - vaddr);

        const size_t block_read = PIN_SafeCopy(
            td->block16.data(), reinterpret_cast<const VOID *>(cur),
            td->block16.size());
        if (block_read < td->block16.size()) {
            std::memset(td->block16.data() + block_read, 0,
                        td->block16.size() - block_read);
        }

        bool ok_paddr = false;
        bool ok_paddr16 = false;
        const uint64_t paddr = VaddrToPaddr(overlap_start, td, &ok_paddr);
        const uint64_t paddr16 = VaddrToPaddr(cur, td, &ok_paddr16);

        uint32_t part_read = 0;
        if (size_read > offset) {
            const size_t remaining = size_read - offset;
            part_read = static_cast<uint32_t>(
                std::min(static_cast<size_t>(overlap_len), remaining));
        }

        RecordHeader hdr;
        hdr.seq = g_seq.fetch_add(1, std::memory_order_relaxed);
        hdr.tid = static_cast<uint32_t>(tid);
        hdr.size = overlap_len;
        hdr.size_read = part_read;
        hdr.block16_read = static_cast<uint32_t>(block_read);
        hdr.vaddr = overlap_start;
        hdr.paddr = ok_paddr ? paddr : 0;
        hdr.paddr16 = ok_paddr16 ? paddr16 : 0;
        hdr.ip = static_cast<uint64_t>(ip);
        hdr.instr_id = ReadInstrId();
        hdr.flags = 0;
        if (ok_paddr) {
            hdr.flags |= kFlagPaddrValid;
        }
        if (ok_paddr16) {
            hdr.flags |= kFlagPaddr16Valid;
        }
        if (!is_after) {
            hdr.flags |= kFlagUsedBefore;
        }
        if (part_read != overlap_len) {
            hdr.flags |= kFlagPartialWriteRead;
        }
        if (block_read != td->block16.size()) {
            hdr.flags |= kFlagPartialBlock16Read;
        }
        const uint32_t stack_depth = CaptureStack(td);
        hdr.stack_depth = stack_depth;

        bool seeded_here = false;
        if (g_seed_from_input_tensor &&
            (td->tensor_assign_depth > 0 || td->update_inputs_depth > 0)) {
            // Seed taint from the model input copy path.
            //
            // The original intent is to mark the post-normalization copy into input
            // placeholders (typically via updateInputPlaceholders -> Tensor::assign).
            // In some builds Tensor::assign may be inlined or use a different symbol,
            // so fall back to seeding any stores executed while inside
            // updateInputPlaceholders*.
            MemSetTaint(overlap_start, overlap_len, true);
            seeded_here = true;
        }

        const bool is_tainted =
            seeded_here ||
            (g_taint_enabled && MemAnyTaint(overlap_start, overlap_len));
        if (g_taint_only && !is_tainted) {
            continue;
        }

        EmitRecord(hdr, td->write_buf.data() + offset, overlap_len,
                   td->block16.data(),
                   stack_depth ? td->stack_ips.data() : nullptr, stack_depth);

        if (is_tainted && g_taint_txt != nullptr) {
            // Format:
            //   seq tid instr_id ip vaddr size size_read after values_kind values_repr values_f32_view
            std::fprintf(g_taint_txt,
                         "%" PRIu64 "\t%u\t%" PRIu64 "\t0x%016" PRIx64
                         "\t0x%016" PRIx64 "\t%u\t%u\t%u\t",
                         hdr.seq, hdr.tid, hdr.instr_id, hdr.ip, hdr.vaddr,
                         hdr.size, hdr.size_read, (is_after ? 1u : 0u));

            const std::string fmt = KnobTaintDecimalFmt.Value();
            const uint8_t *buf = td->write_buf.data() + offset;
            const uint32_t n = hdr.size_read;

            auto print_u8_body = [&]() {
                for (uint32_t i = 0; i < n; ++i) {
                    std::fprintf(g_taint_txt, "%s%u", (i == 0 ? "" : ","),
                                 buf[i]);
                }
            };

            auto print_u16_body = [&]() {
                if ((n % 2) != 0) {
                    print_u8_body();
                    return;
                }
                for (uint32_t i = 0; i < n; i += 2) {
                    uint16_t v = 0;
                    std::memcpy(&v, buf + i, 2);
                    std::fprintf(g_taint_txt, "%s%u", (i == 0 ? "" : ","), v);
                }
            };

            auto print_u32_body = [&]() {
                if ((n % 4) != 0) {
                    print_u8_body();
                    return;
                }
                for (uint32_t i = 0; i < n; i += 4) {
                    uint32_t v = 0;
                    std::memcpy(&v, buf + i, 4);
                    std::fprintf(g_taint_txt, "%s%u", (i == 0 ? "" : ","), v);
                }
            };

            auto print_u64_body = [&]() {
                if ((n % 8) != 0) {
                    print_u8_body();
                    return;
                }
                for (uint32_t i = 0; i < n; i += 8) {
                    uint64_t v = 0;
                    std::memcpy(&v, buf + i, 8);
                    std::fprintf(g_taint_txt, "%s%" PRIu64, (i == 0 ? "" : ","), v);
                }
            };

            auto print_hex_body = [&]() {
                for (uint32_t i = 0; i < n; ++i) {
                    std::fprintf(g_taint_txt, "%02x", buf[i]);
                }
            };

            auto print_f32_body = [&]() {
                if ((n % 4) != 0) {
                    print_u8_body();
                    return;
                }
                for (uint32_t i = 0; i < n; i += 4) {
                    float v = 0.0f;
                    std::memcpy(&v, buf + i, 4);
                    std::fprintf(g_taint_txt, "%s%.9g", (i == 0 ? "" : ","), v);
                }
            };

            auto print_f64_body = [&]() {
                if ((n % 8) != 0) {
                    print_u8_body();
                    return;
                }
                for (uint32_t i = 0; i < n; i += 8) {
                    double v = 0.0;
                    std::memcpy(&v, buf + i, 8);
                    std::fprintf(g_taint_txt, "%s%.17g", (i == 0 ? "" : ","), v);
                }
            };

            std::string main_kind;
            if (fmt == "u8") {
                main_kind = "u8";
            } else if (fmt == "u16") {
                main_kind = ((n % 2) == 0) ? "u16" : "u8";
            } else if (fmt == "u32") {
                main_kind = ((n % 4) == 0) ? "u32" : "u8";
            } else if (fmt == "u64") {
                main_kind = ((n % 8) == 0) ? "u64" : "u8";
            } else if (fmt == "hex") {
                main_kind = (n == 16) ? "hex16" : "bytes";
            } else if (fmt == "f32") {
                main_kind = ((n % 4) == 0) ? "f32" : "u8";
            } else if (fmt == "f64") {
                main_kind = ((n % 8) == 0) ? "f64" : "u8";
            } else { // auto
                if (n == 1) {
                    main_kind = "u8";
                } else if (n == 2) {
                    main_kind = "u16";
                } else if (n == 4) {
                    main_kind = "u32";
                } else if (n == 8) {
                    main_kind = "u64";
                } else if (n == 16) {
                    main_kind = "hex16";
                } else {
                    main_kind = "bytes";
                }
            }

            std::fputs(main_kind.c_str(), g_taint_txt);
            std::fputc('\t', g_taint_txt);

            if (main_kind == "u8") {
                print_u8_body();
            } else if (main_kind == "u16") {
                print_u16_body();
            } else if (main_kind == "u32") {
                print_u32_body();
            } else if (main_kind == "u64") {
                print_u64_body();
            } else if (main_kind == "f32") {
                print_f32_body();
            } else if (main_kind == "f64") {
                print_f64_body();
            } else {
                print_hex_body();
            }

            std::fputc('\t', g_taint_txt);
            if ((n % 4) == 0 && n > 0) {
                std::fputs("f32:", g_taint_txt);
                print_f32_body();
            }
            std::fputc('\n', g_taint_txt);
        }
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

static VOID RegisterInstInfo(INS ins) {
    const uint64_t ip = static_cast<uint64_t>(INS_Address(ins));

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

static uint64_t ReadInstrId() {
    const ADDRINT addr = g_instr_id_addr.load(std::memory_order_relaxed);
    if (addr != 0) {
        uint64_t val = UINT64_MAX;
        const size_t n = PIN_SafeCopy(&val,
                                      reinterpret_cast<const VOID *>(addr),
                                      sizeof(val));
        if (n == sizeof(val)) {
            return val;
        }
    }
    return g_current_instr_id.load(std::memory_order_relaxed);
}

static bool FindSymbolInElf(const std::string &path, const char *sym_name,
                            uint64_t *value_out, bool *is_pie_out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }

    Elf64_Ehdr eh;
    const ssize_t eh_sz = pread(fd, &eh, sizeof(eh), 0);
    if (eh_sz != static_cast<ssize_t>(sizeof(eh))) {
        close(fd);
        return false;
    }
    if (std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 ||
        eh.e_ident[EI_CLASS] != ELFCLASS64 ||
        eh.e_ident[EI_DATA] != ELFDATA2LSB) {
        close(fd);
        return false;
    }

    const size_t shnum = eh.e_shnum;
    const size_t shentsize = eh.e_shentsize;
    const size_t shdr_bytes = shnum * shentsize;
    std::vector<Elf64_Shdr> shdrs(shnum);
    const ssize_t sh_sz =
        pread(fd, shdrs.data(), shdr_bytes, static_cast<off_t>(eh.e_shoff));
    if (sh_sz != static_cast<ssize_t>(shdr_bytes) || shnum == 0) {
        close(fd);
        return false;
    }

    if (eh.e_shstrndx >= shnum) {
        close(fd);
        return false;
    }
    const Elf64_Shdr &shstr = shdrs[eh.e_shstrndx];
    std::vector<char> shstrtab(shstr.sh_size);
    const ssize_t shstr_sz =
        pread(fd, shstrtab.data(), shstrtab.size(), shstr.sh_offset);
    if (shstr_sz != static_cast<ssize_t>(shstrtab.size())) {
        close(fd);
        return false;
    }

    const Elf64_Shdr *symtab = nullptr;
    const Elf64_Shdr *strtab = nullptr;
    const Elf64_Shdr *dynsym = nullptr;
    const Elf64_Shdr *dynstr = nullptr;

    for (size_t i = 0; i < shnum; ++i) {
        const char *name = shstrtab.data() + shdrs[i].sh_name;
        if (std::strcmp(name, ".symtab") == 0) {
            symtab = &shdrs[i];
        } else if (std::strcmp(name, ".strtab") == 0) {
            strtab = &shdrs[i];
        } else if (std::strcmp(name, ".dynsym") == 0) {
            dynsym = &shdrs[i];
        } else if (std::strcmp(name, ".dynstr") == 0) {
            dynstr = &shdrs[i];
        }
    }

    uint64_t value = 0;
    auto scan_table = [&](const Elf64_Shdr *sym,
                          const Elf64_Shdr *str) -> bool {
        if (!sym || !str || sym->sh_entsize == 0) {
            return false;
        }
        std::vector<char> strtab_buf(str->sh_size);
        const ssize_t str_sz =
            pread(fd, strtab_buf.data(), strtab_buf.size(), str->sh_offset);
        if (str_sz != static_cast<ssize_t>(strtab_buf.size())) {
            return false;
        }
        const size_t count = sym->sh_size / sym->sh_entsize;
        for (size_t i = 0; i < count; ++i) {
            Elf64_Sym symrec;
            const off_t off =
                static_cast<off_t>(sym->sh_offset + i * sym->sh_entsize);
            const ssize_t n = pread(fd, &symrec, sizeof(symrec), off);
            if (n != static_cast<ssize_t>(sizeof(symrec))) {
                return false;
            }
            if (symrec.st_name >= strtab_buf.size()) {
                continue;
            }
            const char *name = strtab_buf.data() + symrec.st_name;
            if (std::strcmp(name, sym_name) == 0) {
                value = symrec.st_value;
                return true;
            }
        }
        return false;
    };

    bool found = scan_table(symtab, strtab);
    if (!found) {
        found = scan_table(dynsym, dynstr);
    }
    if (found) {
        *value_out = value;
        *is_pie_out = (eh.e_type == ET_DYN);
        close(fd);
        return true;
    }

    close(fd);
    return false;
}

static VOID OnSetInstrId(THREADID, ADDRINT id) {
    g_current_instr_id.store(static_cast<uint64_t>(id),
                             std::memory_order_relaxed);
}

static VOID Routine(RTN rtn, VOID *) {
    if (!RTN_Valid(rtn)) {
        return;
    }
    const std::string &name = RTN_Name(rtn);
    if (name.find("glow_set_current_instr_id") != std::string::npos) {
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(OnSetInstrId),
                       IARG_THREAD_ID, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_END);
        RTN_Close(rtn);
        return;
    }

    if (!g_seed_from_input_tensor) {
        return;
    }

    // Seed mode "input-tensor": mark the copy into input placeholders as taint
    // source. This happens via updateInputPlaceholders -> Tensor::assign.
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
        return;
    }
}

static VOID ImageLoad(IMG img, VOID *) {
    if (!IMG_Valid(img)) {
        return;
    }
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym)) {
        const std::string name = SYM_Name(sym);
        if (name == "glow_current_instr_id" ||
            name == "_glow_current_instr_id") {
            ADDRINT addr = SYM_Address(sym);
            if (addr == 0) {
                addr = IMG_LowAddress(img) + SYM_Value(sym);
            }
            if (addr != 0) {
                g_instr_id_addr.store(addr, std::memory_order_relaxed);
            }
            break;
        }
    }
    if (g_instr_id_addr.load(std::memory_order_relaxed) == 0 &&
        IMG_IsMainExecutable(img)) {
        uint64_t sym_val = 0;
        bool is_pie = false;
        if (FindSymbolInElf(IMG_Name(img), "glow_current_instr_id", &sym_val,
                            &is_pie)) {
            ADDRINT addr = static_cast<ADDRINT>(sym_val);
            if (is_pie) {
                addr += IMG_LowAddress(img);
            }
            if (addr != 0) {
                g_instr_id_addr.store(addr, std::memory_order_relaxed);
            }
        }
    }
}

static VOID Instruction(INS ins, VOID *) {
    if (INS_IsCall(ins)) {
        if (INS_IsValidForIpointAfter(ins)) {
            INS_InsertCall(ins, IPOINT_AFTER, AFUNPTR(OnCall), IARG_THREAD_ID,
                           IARG_INST_PTR, IARG_END);
        } else {
            INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(OnCall), IARG_THREAD_ID,
                           IARG_INST_PTR, IARG_END);
        }
    } else if (INS_IsRet(ins)) {
        if (INS_IsValidForIpointAfter(ins)) {
            INS_InsertCall(ins, IPOINT_AFTER, AFUNPTR(OnRet), IARG_THREAD_ID,
                           IARG_END);
        } else {
            INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(OnRet), IARG_THREAD_ID,
                           IARG_END);
        }
    }

    const uint64_t ins_ip = static_cast<uint64_t>(INS_Address(ins));
    if (g_has_site_ip_filter && ins_ip != g_site_ip_filter) {
        return;
    }

    const UINT32 mem_ops = INS_MemoryOperandCount(ins);
    if (INS_HasScatteredMemoryAccess(ins)) {
        return;
    }

    // --- Taint propagation (optional) ---
    // Conservative, dataflow-like taint:
    //   taint_out = OR(taint(src_regs), taint(mem_reads))
    //   write regs/mem get taint_out
    //
    // This is designed to be sound-ish (avoid false negatives), not perfectly precise.
    if (g_taint_enabled) {
        bool has_mem_read = false;
        bool has_mem_write = false;
        for (UINT32 mem_op = 0; mem_op < mem_ops; ++mem_op) {
            if (INS_MemoryOperandIsRead(ins, mem_op)) {
                has_mem_read = true;
            }
            if (INS_MemoryOperandIsWritten(ins, mem_op)) {
                has_mem_write = true;
            }
        }

        const bool has_reg_writes = (INS_MaxNumWRegs(ins) > 0);
        if (has_reg_writes || has_mem_write) {
            INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintBegin),
                           IARG_THREAD_ID, IARG_END);

            // Exclude address-generation regs (base/index) from taint sources when possible.
            std::unordered_set<REG> addr_regs;
            addr_regs.reserve(2);
            // Pin's API exposes only one base/index pair per instruction.
            const REG b = NormReg(INS_MemoryBaseReg(ins));
            const REG i = NormReg(INS_MemoryIndexReg(ins));
            if (b != REG_INVALID()) {
                addr_regs.insert(b);
            }
            if (i != REG_INVALID()) {
                addr_regs.insert(i);
            }

            const UINT32 max_r = INS_MaxNumRRegs(ins);
            for (UINT32 i = 0; i < max_r; ++i) {
                REG r = NormReg(INS_RegR(ins, i));
                if (r == REG_INVALID()) {
                    continue;
                }
                if (addr_regs.find(r) != addr_regs.end()) {
                    continue;
                }
                INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccReg),
                               IARG_THREAD_ID, IARG_UINT32,
                               static_cast<UINT32>(r), IARG_END);
            }

            if (has_mem_read) {
                for (UINT32 mem_op = 0; mem_op < mem_ops; ++mem_op) {
                    if (!INS_MemoryOperandIsRead(ins, mem_op)) {
                        continue;
                    }
                    const UINT32 mem_size =
                        static_cast<UINT32>(INS_MemoryOperandSize(ins, mem_op));
                    if (mem_size == 0) {
                        continue;
                    }
                    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintAccMem),
                                   IARG_THREAD_ID, IARG_MEMORYOP_EA, mem_op,
                                   IARG_UINT32, mem_size, IARG_END);
                }
            }

            const UINT32 max_w = INS_MaxNumWRegs(ins);
            for (UINT32 i = 0; i < max_w; ++i) {
                REG r = NormReg(INS_RegW(ins, i));
                if (r == REG_INVALID()) {
                    continue;
                }
                INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetReg),
                               IARG_THREAD_ID, IARG_UINT32,
                               static_cast<UINT32>(r), IARG_END);
            }

            if (has_mem_write) {
                for (UINT32 mem_op = 0; mem_op < mem_ops; ++mem_op) {
                    if (!INS_MemoryOperandIsWritten(ins, mem_op)) {
                        continue;
                    }
                    const UINT32 mem_size =
                        static_cast<UINT32>(INS_MemoryOperandSize(ins, mem_op));
                    if (mem_size == 0) {
                        continue;
                    }
                    INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(TaintSetMem),
                                   IARG_THREAD_ID, IARG_MEMORYOP_EA, mem_op,
                                   IARG_UINT32, mem_size, IARG_END);
                }
            }
        }
    }

    if (mem_ops == 0) {
        return;
    }

    const BOOL has_after = INS_IsValidForIpointAfter(ins);
    const bool snapshot_ea_before =
        // Stack writes (push/call/...) update RSP as part of the instruction,
        // so IARG_MEMORYOP_EA at IPOINT_AFTER can be off by one word.
        INS_IsStackWrite(ins) ||
        // String ops (rep movs/stos/...) update index/count registers during
        // execution; snapshot the EA before execution.
        INS_IsStringop(ins);
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
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, AFUNPTR(RecordInitIfFirst), IARG_THREAD_ID,
            IARG_MEMORYOP_EA, mem_op, IARG_UINT32, mem_size, IARG_INST_PTR,
            IARG_END);
        if (has_after) {
            // Always capture the pre-write payload for the first tainted write to
            // each (vaddr,size,size_read) key.
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, AFUNPTR(RecordFirstTaintBefore),
                IARG_THREAD_ID, IARG_MEMORYOP_EA, mem_op, IARG_UINT32, mem_size,
                IARG_INST_PTR, IARG_END);
            if (snapshot_ea_before) {
                INS_InsertPredicatedCall(
                    ins, IPOINT_BEFORE, AFUNPTR(SaveWriteEA), IARG_THREAD_ID,
                    IARG_UINT32, mem_op, IARG_MEMORYOP_EA, mem_op,
                    IARG_UINT32, mem_size, IARG_END);
                INS_InsertPredicatedCall(
                    ins, IPOINT_AFTER, AFUNPTR(RecordWriteSaved), IARG_THREAD_ID,
                    IARG_UINT32, mem_op, IARG_BOOL, TRUE, IARG_INST_PTR,
                    IARG_END);
            } else {
                INS_InsertPredicatedCall(
                    ins, IPOINT_AFTER, AFUNPTR(RecordWrite), IARG_THREAD_ID,
                    IARG_MEMORYOP_EA, mem_op, IARG_UINT32, mem_size,
                    IARG_BOOL, TRUE, IARG_INST_PTR, IARG_END);
            }
        } else {
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, AFUNPTR(RecordWrite), IARG_THREAD_ID,
                IARG_MEMORYOP_EA, mem_op, IARG_UINT32, mem_size,
                IARG_BOOL, FALSE, IARG_INST_PTR, IARG_END);
        }
    }
}

static VOID ThreadStart(THREADID tid, CONTEXT *, INT32, VOID *) {
    ThreadData *td = new ThreadData();
    td->write_buf.reserve(64);
    td->reg_taint.resize(static_cast<size_t>(REG_LAST));
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
    if (g_taint_txt != nullptr) {
        std::fflush(g_taint_txt);
        std::fclose(g_taint_txt);
        g_taint_txt = nullptr;
    }
    if (g_pagemap_fd >= 0) {
        close(g_pagemap_fd);
        g_pagemap_fd = -1;
    }

    for (auto &kv : g_shadow_pages) {
        delete kv.second;
    }
    g_shadow_pages.clear();

    if (!g_inst_map.empty()) {
        FILE *map = std::fopen(KnobIpMap.Value().c_str(), "w");
        if (map != nullptr) {
            std::fprintf(map,
                         "# ip\timage\timage_offset\troutine\tdisasm\n");
            for (const auto &kv : g_inst_map) {
                std::fprintf(map, "%s\n", kv.second.c_str());
            }
            std::fflush(map);
            std::fclose(map);
        }
    }
}

} // namespace

int main(int argc, char *argv[]) {
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) {
        return Usage();
    }

    g_page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (g_page_size == 0) {
        std::fprintf(stderr, "Failed to get page size.\n");
        return 1;
    }

    g_no_paddr = (KnobNoPaddr.Value() != 0);
    if (!g_no_paddr) {
        g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
        if (g_pagemap_fd < 0) {
            std::perror("open /proc/self/pagemap");
            std::fprintf(stderr, "Run with sudo to read pagemap, or use -no-paddr 1.\n");
            return 1;
        }
    }

    g_out = std::fopen(KnobOutput.Value().c_str(), "wb");
    if (g_out == nullptr) {
        std::perror("fopen trace output");
        return 1;
    }

    FileHeader fh;
    std::memcpy(fh.magic, "PADDRTRC", sizeof(fh.magic));
    fh.version = 4;
    fh.reserved = 0;
    fh.page_size = static_cast<uint64_t>(g_page_size);
    std::fwrite(&fh, sizeof(fh), 1, g_out);

    PIN_InitLock(&g_lock);
    PIN_InitLock(&g_seen_lock);
    PIN_InitLock(&g_inst_lock);
    PIN_InitLock(&g_taint_lock);
    g_tls_key = PIN_CreateThreadDataKey(nullptr);

    g_taint_file = KnobTaintFile.Value();
    g_taint_seed_mode = KnobTaintSeedMode.Value();
    g_seed_from_input_tensor = (g_taint_seed_mode == "input-tensor");
    g_taint_enabled = (!g_taint_file.empty()) || g_seed_from_input_tensor;
    g_taint_only = (KnobTaintOnly.Value() != 0);
    g_taint_no_lock = (KnobTaintNoLock.Value() != 0);
    g_after_only = (KnobAfterOnly.Value() != 0);
    if (!KnobSiteIp.Value().empty()) {
        if (!ParseUint64Arg(KnobSiteIp.Value(), &g_site_ip_filter)) {
            std::fprintf(stderr, "invalid -site-ip: %s\n",
                         KnobSiteIp.Value().c_str());
            return 1;
        }
        g_has_site_ip_filter = true;
    }
    if (!KnobStackContainsIp.Value().empty()) {
        if (!ParseUint64Arg(KnobStackContainsIp.Value(),
                            &g_stack_contains_ip_filter)) {
            std::fprintf(stderr, "invalid -stack-contains-ip: %s\n",
                         KnobStackContainsIp.Value().c_str());
            return 1;
        }
        g_has_stack_contains_ip_filter = true;
    }
    if (g_taint_enabled) {
        if (!g_taint_file.empty()) {
            g_taint_file_base = BaseName(g_taint_file);
        }
        const std::string out_txt = KnobTaintDecimalOut.Value();
        if (!out_txt.empty()) {
            g_taint_txt = std::fopen(out_txt.c_str(), "w");
            if (g_taint_txt == nullptr) {
                std::perror("fopen taint decimal output");
                return 1;
            }
            std::fprintf(
                g_taint_txt,
                "# seq\ttid\tinstr_id\tip\tvaddr\tsize\tsize_read\tafter\tvalues_kind\tvalues_repr\tvalues_f32_view\n");
            std::fflush(g_taint_txt);
        }
    }

    IMG_AddInstrumentFunction(ImageLoad, nullptr);
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
