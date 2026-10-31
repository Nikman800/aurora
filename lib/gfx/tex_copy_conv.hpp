#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

#include <array>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct alignas(16) Uniforms {
  Vec2<float> offset;
  Vec2<float> scale{1.f, 1.f};
  uint32_t opaqueAlpha = 0;
  // Copy vertical filter: the rows above and below (rowStep apart in UV) and the row itself,
  // weighted by vfWeights.xyz (sum 1 for the identity filter); UV y is clamped to clampY.
  uint32_t vfEnable = 0;
  float rowStep = 0.f;
  uint32_t _pad0 = 0;
  std::array<float, 4> vfWeights{0.f, 1.f, 0.f, 0.f};
  Vec2<float> clampY{0.f, 1.f};
  std::array<uint32_t, 2> _pad1{};
};
static_assert(sizeof(Uniforms) == 64);

struct ConvRequest {
  GXTexFmt fmt;
  GXPixelFmt srcFmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // Uniforms
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);

bool snapshot_depth_supported() noexcept;
void snapshot_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& srcDepth, uint32_t msaaSamples,
                    const wgpu::TextureView& dst);

} // namespace aurora::gfx::tex_copy_conv
