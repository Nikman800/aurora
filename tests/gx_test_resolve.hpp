#pragma once

// What the test stub of gfx::resolve_pass_into last received (an EFB copy).
#include "gfx/recording.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gfx::testing {
struct LastResolve {
  ClipRect rect{};
  bool clearColor = false;
  bool clearAlpha = false;
  bool clearDepth = false;
  GXTexFmt format = GX_TF_I4;
  GXPixelFmt sourceFormat = GX_PF_RGB8_Z24;
  CopyFilter filter{};
};
extern LastResolve lastResolve;
extern std::atomic<uint32_t> keepEfbCount;
} // namespace aurora::gfx::testing
