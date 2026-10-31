#pragma once

#include "../internal.hpp"

#include <dolphin/gx/GXAurora.h>

#include <cstdint>
#include <span>
#include <utility>

// GX readback frontend
namespace aurora::gx::readback {

inline constexpr u32 MaxCaptureItems = 4096;
inline constexpr u32 MaxPayloadBytes = 32u << 20;

GXBool capture_peeks(AuroraReadbackStream stream, const AuroraReadbackPeekRequest* requests, u32 count,
                     GXAlphaReadMode alphaMode);
GXBool capture_copies(AuroraReadbackStream stream, const AuroraReadbackCopySrc* srcs, u32 count);
GXBool get(AuroraReadbackStream stream, AuroraReadbackDest* dests, u32 count);

// Command processor: GX_AURORA_READBACK_CAPTURE_PEEKS and GX_AURORA_READBACK_CAPTURE_COPIES
void handle_capture_peeks(ByteReader& reader);
void handle_capture_copies(ByteReader& reader);
// Records a capture of texture copies (key, copy destination) into `stream` at this point of the frame.
bool record_copy_capture(AuroraReadbackStream stream, uint64_t sequence,
                         std::span<const std::pair<uint64_t, const void*>> sources);

} // namespace aurora::gx::readback
