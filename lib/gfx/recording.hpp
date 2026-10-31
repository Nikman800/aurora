#pragma once

#include "types.hpp"
#include "../imgui.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace aurora::gfx::detail {

struct FramePacket;

struct RecordedFrame {
  FramePacket* packet;
  size_t frameSlot;
};

void begin_recording(FramePacket& packet, size_t frameSlot);
void prepare_recording();
void record_xfb_present(imgui::DrawData drawData, wgpu::BindGroup rmlBindGroup, bool rmlOverlay);
RecordedFrame end_recording();
void shutdown_recording();
void increment_merged_draw_count() noexcept;

namespace testing {
struct OffscreenCacheStats {
  uint64_t bytes;
  size_t entries;
};
void suppress_render_worker(bool suppress) noexcept;
void seed_offscreen_cache(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                          wgpu::TextureFormat depthFormat);
OffscreenCacheStats offscreen_cache_stats() noexcept;
bool is_offscreen_cached(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                         wgpu::TextureFormat depthFormat);
webgpu::TextureWithSampler recorded_present_source(PresentTargetId target);
} // namespace testing

} // namespace aurora::gfx::detail

namespace aurora::gfx::readback {
struct Job;
}

namespace aurora::gfx {
struct ColorPassDescriptor {
  const char* label = nullptr;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  wgpu::TextureView depthStencilView;
  wgpu::TextureFormat depthStencilFormat = wgpu::TextureFormat::Undefined;
  wgpu::Extent3D targetSize;
  uint32_t sampleCount = 1;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Clear;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::Color clearColor{0.f, 0.f, 0.f, 0.f};
  bool hasDepth = false;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Undefined;
  float depthClearValue = 0.f;
  bool hasStencil = false;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
};

void finish();
void begin_color_pass(const ColorPassDescriptor& desc);
void end_color_pass();
void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size);
void begin_offscreen(uint32_t width, uint32_t height);
void end_offscreen();
bool has_normal_attachment() noexcept;
RenderTargetLayout get_render_target_layout() noexcept;
namespace tex_palette_conv {
struct ConvRequest;
}
void queue_palette_conv(tex_palette_conv::ConvRequest req);

Range push_verts(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_verts(ArrayRef<T> data, size_t alignment) {
  return push_verts(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_indices(ArrayRef<T> data, size_t alignment) {
  return push_indices(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_uniform(const uint8_t* data, size_t length);
template <typename T>
Range push_uniform(const T& data) {
  return push_uniform(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_storage(const uint8_t* data, size_t length);
template <typename T>
Range push_storage(ArrayRef<T> data) {
  return push_storage(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}
template <typename T>
Range push_storage(const T& data) {
  return push_storage(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_texture_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage);

template <typename DrawData>
void push_draw_command(DrawData data);
template <typename DrawData>
DrawData* get_last_draw_command();
template <typename PipelineConfig>
PipelineRef pipeline_ref(const PipelineConfig& config);

// Vertical filter of an EFB copy: weights of the row above, the row and the row below (the
// hardware's seven coefficients summed per row, over 64), and whether the sampled rows are
// clamped to the top and bottom of the source rectangle.
struct CopyFilter {
  bool enabled = false;
  std::array<float, 3> weights{0.f, 1.f, 0.f};
  bool clampTop = true;
  bool clampBottom = true;
};

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat, const CopyFilter& filter = {});
uint32_t align_uniform(uint32_t value);
// The next frame starts with this frame's EFB contents (a display copy that did not clear it).
void keep_efb_next_frame() noexcept;
// The part of the EFB (render target pixels) the last recorded frame presents: the VI's picture size
// (vi::configured_picture_size) from the top left.
ClipRect presented_efb_rect(uint32_t efbWidth, uint32_t efbHeight) noexcept;
// Waits until everything recorded so far is submitted to the GPU; then readback buffers complete
// as poll_gpu_events processes the device's events.
void finish_gpu_work() noexcept;
void poll_gpu_events() noexcept;
Vec2<uint32_t> get_render_target_size() noexcept;
void set_viewport(const Viewport& viewport) noexcept;
void set_scissor(const ClipRect& scissor) noexcept;
void push_debug_group(std::string label);
void insert_debug_marker(std::string label);

struct ReadbackCapture {
  bool active = false;
  bool offscreen = false;
  uint32_t width = 0;
  uint32_t height = 0;
};
ReadbackCapture readback_capture() noexcept;
bool push_readback_op(std::shared_ptr<readback::Job> job, bool efbSource);
} // namespace aurora::gfx
