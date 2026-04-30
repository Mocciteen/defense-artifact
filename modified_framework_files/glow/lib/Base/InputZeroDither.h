#ifndef GLOW_LIB_BASE_INPUTZERODITHER_H
#define GLOW_LIB_BASE_INPUTZERODITHER_H

#include "glow/Base/Tensor.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
namespace glow {

enum class InputZeroDitherMode {
  Checkerboard,
  Random,
};

enum class InputZeroDitherLayout {
  NCHW,
  NHWC,
};

struct InputZeroDitherConfig {
  bool enabled{false};
  InputZeroDitherMode mode{InputZeroDitherMode::Checkerboard};
  float epsMin{1.0e-8f};
  float epsMax{2.0e-8f};
  float zeroThreshold{0.1f};
  InputZeroDitherLayout layout{InputZeroDitherLayout::NCHW};
  bool hasSeed{false};
  uint32_t seed{0u};
  bool silent{false};
};

inline bool inputZeroDitherIEq(const char *lhs, const char *rhs) {
  if (!lhs || !rhs) {
    return lhs == rhs;
  }
  while (*lhs && *rhs) {
    const unsigned char lc = static_cast<unsigned char>(*lhs);
    const unsigned char rc = static_cast<unsigned char>(*rhs);
    const unsigned char lLower =
        (lc >= 'A' && lc <= 'Z') ? static_cast<unsigned char>(lc - 'A' + 'a')
                                 : lc;
    const unsigned char rLower =
        (rc >= 'A' && rc <= 'Z') ? static_cast<unsigned char>(rc - 'A' + 'a')
                                 : rc;
    if (lLower != rLower) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

inline bool parseInputZeroDitherBoolEnv(const char *name) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return false;
  }
  if (inputZeroDitherIEq(value, "0") || inputZeroDitherIEq(value, "off") ||
      inputZeroDitherIEq(value, "false") || inputZeroDitherIEq(value, "no")) {
    return false;
  }
  return true;
}

inline bool parseInputZeroDitherFloatEnvOrExit(const char *name, float *out) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return false;
  }

  char *end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (!end || *end != '\0' || !std::isfinite(parsed)) {
    std::cerr << "invalid " << name << "=" << value << "\n";
    std::exit(2);
  }
  *out = parsed;
  return true;
}

inline bool parseInputZeroDitherUint32EnvOrExit(const char *name,
                                                uint32_t *out) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return false;
  }

  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 0);
  if (!end || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    std::cerr << "invalid " << name << "=" << value << "\n";
    std::exit(2);
  }
  *out = static_cast<uint32_t>(parsed);
  return true;
}

inline InputZeroDitherLayout parseInputZeroDitherLayoutOrExit(
    const char *value) {
  if (inputZeroDitherIEq(value, "NCHW")) {
    return InputZeroDitherLayout::NCHW;
  }
  if (inputZeroDitherIEq(value, "NHWC")) {
    return InputZeroDitherLayout::NHWC;
  }
  std::cerr << "invalid GLOW_INPUT_ZERO_DITHER_LAYOUT=" << value << "\n";
  std::exit(2);
}

inline InputZeroDitherConfig loadInputZeroDitherConfigOrExit(
    InputZeroDitherLayout defaultLayout) {
  InputZeroDitherConfig config;
  config.layout = defaultLayout;
  config.silent = parseInputZeroDitherBoolEnv("GLOW_INPUT_ZERO_DITHER_SILENT");

  const char *mode = std::getenv("GLOW_INPUT_ZERO_DITHER");
  if (!mode || mode[0] == '\0' || inputZeroDitherIEq(mode, "0") ||
      inputZeroDitherIEq(mode, "off") || inputZeroDitherIEq(mode, "none")) {
    return config;
  }

  if (inputZeroDitherIEq(mode, "checker") ||
      inputZeroDitherIEq(mode, "checkerboard")) {
    config.mode = InputZeroDitherMode::Checkerboard;
  } else if (inputZeroDitherIEq(mode, "random") ||
             inputZeroDitherIEq(mode, "rand")) {
    config.mode = InputZeroDitherMode::Random;
  } else {
    std::cerr << "invalid GLOW_INPUT_ZERO_DITHER=" << mode << "\n";
    std::exit(2);
  }

  config.enabled = true;
  parseInputZeroDitherFloatEnvOrExit("GLOW_INPUT_ZERO_DITHER_EPS0",
                                     &config.epsMin);
  parseInputZeroDitherFloatEnvOrExit("GLOW_INPUT_ZERO_DITHER_EPS1",
                                     &config.epsMax);
  parseInputZeroDitherFloatEnvOrExit("GLOW_INPUT_ZERO_DITHER_EPS_MIN",
                                     &config.epsMin);
  parseInputZeroDitherFloatEnvOrExit("GLOW_INPUT_ZERO_DITHER_EPS_MAX",
                                     &config.epsMax);
  parseInputZeroDitherFloatEnvOrExit("GLOW_INPUT_ZERO_DITHER_THRESH",
                                     &config.zeroThreshold);
  config.hasSeed = parseInputZeroDitherUint32EnvOrExit(
      "GLOW_INPUT_ZERO_DITHER_SEED", &config.seed);

  const char *layout = std::getenv("GLOW_INPUT_ZERO_DITHER_LAYOUT");
  if (layout && layout[0] != '\0') {
    config.layout = parseInputZeroDitherLayoutOrExit(layout);
  }

  if (config.zeroThreshold < 0.0f) {
    std::cerr << "GLOW_INPUT_ZERO_DITHER_THRESH must be >= 0\n";
    std::exit(2);
  }
  if (config.epsMin < 0.0f || config.epsMax < 0.0f) {
    std::cerr << "GLOW_INPUT_ZERO_DITHER_EPS ranges must be >= 0\n";
    std::exit(2);
  }
  if (config.epsMin > config.epsMax) {
    std::cerr << "GLOW_INPUT_ZERO_DITHER_EPS_MIN must be <= "
                 "GLOW_INPUT_ZERO_DITHER_EPS_MAX\n";
    std::exit(2);
  }

  return config;
}

inline uint32_t mixInputZeroDitherHash(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

inline uint32_t makeInputZeroDitherHashSeed(const InputZeroDitherConfig &config,
                                            size_t linearIndex, size_t y,
                                            size_t x) {
  const uint64_t index64 = static_cast<uint64_t>(linearIndex);
  uint32_t seed = config.hasSeed ? config.seed : 0x6d2b79f5U;
  seed ^= static_cast<uint32_t>(index64);
  seed ^= static_cast<uint32_t>(index64 >> 32) * 0x9e3779b9U;
  seed ^= static_cast<uint32_t>(y) * 0x85ebca6bU;
  seed ^= static_cast<uint32_t>(x) * 0xc2b2ae35U;
  return mixInputZeroDitherHash(seed);
}

inline float computeInputZeroDitherDelta(const InputZeroDitherConfig &config,
                                         size_t linearIndex, size_t y,
                                         size_t x) {
  if (config.mode == InputZeroDitherMode::Checkerboard) {
    return ((y + x) & 1U) == 0 ? config.epsMin : config.epsMax;
  }

  const uint32_t hash =
      makeInputZeroDitherHashSeed(config, linearIndex, y, x);
  const float unit =
      static_cast<float>(hash >> 8) * (1.0f / 16777216.0f);
  return config.epsMin + (config.epsMax - config.epsMin) * unit;
}

inline float ditherInputZeroValue(float value, size_t linearIndex, size_t y,
                                  size_t x,
                                  const InputZeroDitherConfig &config,
                                  size_t *changed) {
  if (!config.enabled || std::fabs(value) > config.zeroThreshold) {
    return value;
  }

  value += computeInputZeroDitherDelta(config, linearIndex, y, x);

  if (changed) {
    ++(*changed);
  }
  return value;
}

inline void logInputZeroDitherSummary(const InputZeroDitherConfig &config,
                                      size_t changed, size_t total) {
  if (config.silent) {
    return;
  }

  std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10);
  if (config.mode == InputZeroDitherMode::Checkerboard) {
    std::cerr << "input_zero_dither=checkerboard"
              << " layout="
              << (config.layout == InputZeroDitherLayout::NCHW ? "NCHW"
                                                               : "NHWC")
              << " eps0=" << config.epsMin << " eps1=" << config.epsMax
              << " thresh=" << config.zeroThreshold << " changed=" << changed
              << "/" << total;
  } else {
    std::cerr << "input_zero_dither=random"
              << " layout="
              << (config.layout == InputZeroDitherLayout::NCHW ? "NCHW"
                                                               : "NHWC")
              << " eps_min=" << config.epsMin << " eps_max=" << config.epsMax
              << " thresh=" << config.zeroThreshold << " changed=" << changed
              << "/" << total;
    if (config.hasSeed) {
      std::cerr << " seed=" << config.seed;
    } else {
      std::cerr << " seed=default_hash";
    }
  }
  std::cerr << "\n";
}

inline size_t applyInputZeroDitherOrExit(Tensor &inputImageData,
                                         InputZeroDitherLayout actualLayout) {
  const InputZeroDitherConfig config =
      loadInputZeroDitherConfigOrExit(actualLayout);
  if (!config.enabled) {
    return 0;
  }

  assert(inputImageData.getElementType() == ElemKind::FloatTy);
  const auto dims = inputImageData.dims();
  assert(dims.size() == 4 && "input zero dither expects 4D input");

  const size_t n = dims[0];
  const size_t c =
      actualLayout == InputZeroDitherLayout::NCHW ? dims[1] : dims[3];
  const size_t h =
      actualLayout == InputZeroDitherLayout::NCHW ? dims[2] : dims[1];
  const size_t w =
      actualLayout == InputZeroDitherLayout::NCHW ? dims[3] : dims[2];

  auto handle = inputImageData.getHandle<float>();
  size_t changed = 0;

  if (config.layout == InputZeroDitherLayout::NCHW) {
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t ci = 0; ci < c; ++ci) {
        for (size_t yi = 0; yi < h; ++yi) {
          for (size_t xi = 0; xi < w; ++xi) {
            const size_t index = ((ni * c + ci) * h + yi) * w + xi;
            handle.raw(index) = ditherInputZeroValue(
                handle.raw(index), index, yi, xi, config, &changed);
          }
        }
      }
    }
  } else {
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t yi = 0; yi < h; ++yi) {
        for (size_t xi = 0; xi < w; ++xi) {
          for (size_t ci = 0; ci < c; ++ci) {
            const size_t index = ((ni * h + yi) * w + xi) * c + ci;
            handle.raw(index) = ditherInputZeroValue(
                handle.raw(index), index, yi, xi, config, &changed);
          }
        }
      }
    }
  }

  logInputZeroDitherSummary(config, changed, n * c * h * w);
  return changed;
}

} // namespace glow

#endif
