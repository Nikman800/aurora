#pragma once

#include <dolphin/gx/GXStruct.h>

#include <cstdint>

#include <aurora/math.hpp>

namespace aurora::vi {
void configure(const GXRenderModeObj* rm) noexcept;
Vec2<uint32_t> configured_fb_size() noexcept;
// The part of the EFB the picture shows: the frame buffer width and the XFB height when that is
// smaller than the EFB height (the EFB lines below the picture are never shown).
Vec2<uint32_t> configured_picture_size() noexcept;
} // namespace aurora::vi
