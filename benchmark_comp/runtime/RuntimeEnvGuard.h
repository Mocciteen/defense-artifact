#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace benchmark_runtime {

inline bool iequals(const char *lhs, const char *rhs) {
  if (!lhs || !rhs) {
    return lhs == rhs;
  }
  while (*lhs && *rhs) {
    unsigned char a = static_cast<unsigned char>(*lhs);
    unsigned char b = static_cast<unsigned char>(*rhs);
    if (a >= 'A' && a <= 'Z') {
      a = static_cast<unsigned char>(a - 'A' + 'a');
    }
    if (b >= 'A' && b <= 'Z') {
      b = static_cast<unsigned char>(b - 'A' + 'a');
    }
    if (a != b) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

[[noreturn]] inline void failRequirement(const char *name,
                                         const std::string &message) {
  std::cerr << "runtime guard failed for " << name << ": " << message << "\n";
  std::exit(2);
}

inline void requireEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !(iequals(value, "1") || iequals(value, "true") ||
                  iequals(value, "on") || iequals(value, "yes"))) {
    failRequirement(name, "must be enabled");
  }
}

inline void requirePresent(const char *name) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    failRequirement(name, "missing");
  }
}

inline void requireEnvironmentList(const char *listName) {
  const char *raw = std::getenv(listName);
  if (!raw || raw[0] == '\0') {
    failRequirement(listName, "missing required environment list");
  }
  std::stringstream stream(raw);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const auto begin = item.find_first_not_of(" \t");
    const auto end = item.find_last_not_of(" \t");
    if (begin == std::string::npos) {
      continue;
    }
    requirePresent(item.substr(begin, end - begin + 1).c_str());
  }
}

inline void requireStringEquals(const char *name, const char *expected) {
  const char *value = std::getenv(name);
  if (!value || !iequals(value, expected)) {
    failRequirement(name, std::string("expected ") + expected);
  }
}

inline void requireUintEquals(const char *name, unsigned long long expected) {
  const char *value = std::getenv(name);
  if (!value) {
    failRequirement(name, "missing");
  }
  char *end = nullptr;
  unsigned long long actual = std::strtoull(value, &end, 0);
  if (!end || *end || actual != expected) {
    failRequirement(name, "unexpected integer value");
  }
}

inline void requireFloatEquals(const char *name, double expected) {
  const char *value = std::getenv(name);
  if (!value) {
    failRequirement(name, "missing");
  }
  char *end = nullptr;
  double actual = std::strtod(value, &end);
  if (!end || *end || !std::isfinite(actual) ||
      std::fabs(actual - expected) > std::max(1.0e-12, std::fabs(expected) * 1.0e-6)) {
    failRequirement(name, "unexpected float value");
  }
}

} // namespace benchmark_runtime
