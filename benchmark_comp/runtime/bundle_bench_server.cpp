#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "FixedAddressArena.h"
#include "RuntimeEnvGuard.h"
#include "cipherfix_taint_api.h"
#if defined(ZEBRAFIX_PROTECT_INPUT_CLOSURE)
#include "ZebrafixRuntime.h"
#endif

#ifndef BUNDLE_HEADER
#error "Define BUNDLE_HEADER as a quoted generated bundle header."
#endif
#include BUNDLE_HEADER

#ifndef BUNDLE_ENTRY
#error "Define BUNDLE_ENTRY."
#endif
#ifndef BUNDLE_CONSTANT_MEM_SIZE
#error "Define BUNDLE_CONSTANT_MEM_SIZE."
#endif
#ifndef BUNDLE_MUTABLE_MEM_SIZE
#error "Define BUNDLE_MUTABLE_MEM_SIZE."
#endif
#ifndef BUNDLE_ACTIVATIONS_MEM_SIZE
#error "Define BUNDLE_ACTIVATIONS_MEM_SIZE."
#endif
#ifndef BUNDLE_MEM_ALIGN
#error "Define BUNDLE_MEM_ALIGN."
#endif
#ifndef BUNDLE_DATA_OFFSET
#error "Define BUNDLE_DATA_OFFSET."
#endif
#ifndef BUNDLE_OUTPUT_OFFSET
#error "Define BUNDLE_OUTPUT_OFFSET."
#endif
#ifndef BUNDLE_INPUT_BYTES
#error "Define BUNDLE_INPUT_BYTES."
#endif
#ifndef BUNDLE_OUTPUT_ELEMENTS
#define BUNDLE_OUTPUT_ELEMENTS 10
#endif
#ifndef BUNDLE_NAME
#define BUNDLE_NAME "bundle"
#endif

#if defined(BUNDLE_USE_ZEBRA_ENTRY)
extern "C" int BUNDLE_ENTRY_ZEBRA(std::uint8_t *, std::uint8_t *, std::uint8_t *);
#undef BUNDLE_ENTRY
#define BUNDLE_ENTRY BUNDLE_ENTRY_ZEBRA
#endif

namespace {

using benchmark_runtime::FixedAddressArena;

std::uint64_t nowNs() {
  timespec ts {};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    std::cerr << "clock_gettime failed\n";
    std::exit(1);
  }
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

void loadFile(const std::string &path, void *dst, std::size_t expected) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::cerr << "failed to open " << path << "\n";
    std::exit(1);
  }
  input.read(static_cast<char *>(dst), static_cast<std::streamsize>(expected));
  if (static_cast<std::size_t>(input.gcount()) != expected) {
    std::cerr << "unexpected file size for " << path << "\n";
    std::exit(1);
  }
}

std::vector<char> loadInput(const std::string &path) {
  std::vector<char> input(BUNDLE_INPUT_BYTES);
  loadFile(path, input.data(), input.size());
  return input;
}

void verifyRuntimeConfig() {
#if defined(REQUIRE_RUNTIME_ENV_GUARDS)
  benchmark_runtime::requireEnvironmentList("BUNDLE_REQUIRED_ENV");
#endif
}

std::uint64_t hashBytes(const void *data, std::size_t size) {
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct RunResult {
  int rc = 0;
  std::uint64_t stageNs = 0;
  std::uint64_t inferNs = 0;
  std::uint64_t totalNs = 0;
  std::uint64_t outputHash = 0;
  int argmax = 0;
  float output0 = 0.0f;
  std::vector<float> output;
};

RunResult runOnce(std::uint8_t *constantWeight, std::uint8_t *mutableWeight,
                  std::uint8_t *activations,
                  const std::vector<char> &inputTemplate) {
  const std::uint64_t t0 = nowNs();
#if defined(ZEBRAFIX_PROTECT_INPUT_CLOSURE)
  zebrafixClearRanges();
  zebrafixRegisterRange(mutableWeight, BUNDLE_MUTABLE_MEM_SIZE);
  zebrafixRegisterRange(activations, BUNDLE_ACTIVATIONS_MEM_SIZE);
  zebrafixStoreBytes(mutableWeight + BUNDLE_DATA_OFFSET, inputTemplate.data(),
                     inputTemplate.size());
#elif defined(CIPHERFIX_TAINT_INPUT)
  std::memset(mutableWeight, 0, BUNDLE_MUTABLE_MEM_SIZE);
  std::memset(activations, 0, BUNDLE_ACTIVATIONS_MEM_SIZE);
  std::memcpy(mutableWeight + BUNDLE_DATA_OFFSET, inputTemplate.data(),
              inputTemplate.size());
  cipherfixClassifyInput(mutableWeight + BUNDLE_DATA_OFFSET, inputTemplate.size());
#else
  std::memcpy(mutableWeight + BUNDLE_DATA_OFFSET, inputTemplate.data(),
              inputTemplate.size());
#endif
  const std::uint64_t t1 = nowNs();
  const int rc = BUNDLE_ENTRY(constantWeight, mutableWeight, activations);
  const std::uint64_t t2 = nowNs();

  RunResult result;
  result.rc = rc;
  result.stageNs = t1 - t0;
  result.inferNs = t2 - t1;
  result.totalNs = t2 - t0;
  result.output.resize(BUNDLE_OUTPUT_ELEMENTS);
#if defined(ZEBRAFIX_PROTECT_INPUT_CLOSURE)
  zebrafixLoadBytes(result.output.data(), mutableWeight + BUNDLE_OUTPUT_OFFSET,
                    result.output.size() * sizeof(float));
#else
  std::memcpy(result.output.data(), mutableWeight + BUNDLE_OUTPUT_OFFSET,
              result.output.size() * sizeof(float));
  cipherfixDeclassifyBuffer(result.output.data(), result.output.size() * sizeof(float));
  cipherfixDropTaintState();
#endif
  result.outputHash = hashBytes(result.output.data(), result.output.size() * sizeof(float));
  result.argmax = static_cast<int>(std::max_element(result.output.begin(), result.output.end()) - result.output.begin());
  result.output0 = result.output.empty() ? 0.0f : result.output[0];
  return result;
}

void printReady() {
  std::cout << "{\"type\":\"ready\",\"bundle\":\"" << BUNDLE_NAME
            << "\",\"input_bytes\":" << BUNDLE_INPUT_BYTES
            << ",\"output_elements\":" << BUNDLE_OUTPUT_ELEMENTS << "}\n";
  std::cout.flush();
}

void printInfo() {
  std::cout << "{\"type\":\"info\",\"bundle\":\"" << BUNDLE_NAME
            << "\",\"constant_mem_size\":" << BUNDLE_CONSTANT_MEM_SIZE
            << ",\"mutable_mem_size\":" << BUNDLE_MUTABLE_MEM_SIZE
            << ",\"activations_mem_size\":" << BUNDLE_ACTIVATIONS_MEM_SIZE
            << "}\n";
  std::cout.flush();
}

void printResult(const RunResult &r) {
  std::cout << "{\"type\":\"result\",\"rc\":" << r.rc
            << ",\"stage_ns\":" << r.stageNs
            << ",\"infer_ns\":" << r.inferNs
            << ",\"total_ns\":" << r.totalNs
            << ",\"output_hash\":\"0x" << std::hex << r.outputHash << std::dec
            << "\",\"argmax\":" << r.argmax << ",\"output0\":";
  if (std::isfinite(r.output0)) {
    std::cout << r.output0;
  } else {
    std::cout << "null";
  }
  std::cout << "}\n";
  std::cout.flush();
}

bool readExactOrEof(std::istream &input, void *dst, std::size_t expected,
                    bool *sawEof) {
  auto *out = static_cast<char *>(dst);
  std::size_t total = 0;
  *sawEof = false;
  while (total < expected) {
    input.read(out + total, static_cast<std::streamsize>(expected - total));
    const auto got = static_cast<std::size_t>(input.gcount());
    if (got == 0) {
      *sawEof = total == 0 && input.eof();
      return false;
    }
    total += got;
  }
  return true;
}

int serverLoop(std::uint8_t *constantWeight, std::uint8_t *mutableWeight,
               std::uint8_t *activations,
               const std::vector<char> &inputTemplate) {
  printReady();
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "RUN") {
      printResult(runOnce(constantWeight, mutableWeight, activations, inputTemplate));
    } else if (line == "INFO") {
      printInfo();
    } else if (line == "QUIT") {
      std::cout << "{\"type\":\"bye\"}\n";
      return 0;
    }
  }
  return 0;
}

int streamLoop(std::uint8_t *constantWeight, std::uint8_t *mutableWeight,
               std::uint8_t *activations) {
  std::vector<char> input(BUNDLE_INPUT_BYTES);
  while (true) {
    bool sawEof = false;
    if (!readExactOrEof(std::cin, input.data(), input.size(), &sawEof)) {
      return sawEof ? 0 : 1;
    }
    const RunResult result = runOnce(constantWeight, mutableWeight, activations, input);
    std::cout.write(reinterpret_cast<const char *>(result.output.data()),
                    static_cast<std::streamsize>(result.output.size() * sizeof(float)));
    std::cout.flush();
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <weights.bin> <input.bin> [--server|--stream]\n";
    return 2;
  }
  const bool server = argc >= 4 && std::string(argv[3]) == "--server";
  const bool stream = argc >= 4 && std::string(argv[3]) == "--stream";
  verifyRuntimeConfig();
  FixedAddressArena arena(BUNDLE_MEM_ALIGN, BUNDLE_CONSTANT_MEM_SIZE,
                          BUNDLE_MUTABLE_MEM_SIZE, BUNDLE_ACTIVATIONS_MEM_SIZE);
  loadFile(argv[1], arena.constantWeight(), BUNDLE_CONSTANT_MEM_SIZE);
  if (stream) {
    return streamLoop(arena.constantWeight(), arena.mutableWeight(), arena.activations());
  }
  const std::vector<char> input = loadInput(argv[2]);
  if (server) {
    return serverLoop(arena.constantWeight(), arena.mutableWeight(), arena.activations(), input);
  }
  const RunResult result = runOnce(arena.constantWeight(), arena.mutableWeight(), arena.activations(), input);
  printResult(result);
  return result.rc;
}
