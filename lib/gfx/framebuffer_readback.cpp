#include "efb.hpp"
#include "render_worker.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.h>

#include <cstring>

namespace aurora::gfx {
namespace {
Module Log("aurora::gfx::framebuffer_readback");
} // namespace

// Synchronous copy of the EFB color attachment as of the last end_frame. This stalls until the GPU has
// finished all submitted work, so it is meant for tests and screenshots, not per-frame use.
static bool read_framebuffer(void* dst, size_t dstSize, uint32_t* outWidth, uint32_t* outHeight) {
  render_worker::synchronize();
  const auto state = efb::applied_state();
  const auto& source = state.present_source();
  if (!source.texture) {
    Log.error("No framebuffer to read back");
    return false;
  }
  const uint32_t width = source.size.width;
  const uint32_t height = source.size.height;
  if (outWidth != nullptr) {
    *outWidth = width;
  }
  if (outHeight != nullptr) {
    *outHeight = height;
  }
  const bool bgra = source.format == wgpu::TextureFormat::BGRA8Unorm;
  if (!bgra && source.format != wgpu::TextureFormat::RGBA8Unorm) {
    Log.error("Unsupported framebuffer format {}", static_cast<uint32_t>(source.format));
    return false;
  }
  const size_t rowSize = size_t{width} * 4;
  if (dst == nullptr || dstSize < rowSize * height) {
    return false;
  }

  const uint32_t pitch = (static_cast<uint32_t>(rowSize) + 255u) & ~255u;
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Framebuffer readback",
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
      .size = uint64_t{pitch} * height,
  };
  const auto buffer = webgpu::g_device.CreateBuffer(&bufferDescriptor);
  const auto encoder = webgpu::g_device.CreateCommandEncoder();
  const wgpu::TexelCopyTextureInfo src{.texture = source.texture};
  const wgpu::TexelCopyBufferInfo dstInfo{
      .layout = {.offset = 0, .bytesPerRow = pitch, .rowsPerImage = height},
      .buffer = buffer,
  };
  const wgpu::Extent3D extent{width, height, 1};
  encoder.CopyTextureToBuffer(&src, &dstInfo, &extent);
  const auto commands = encoder.Finish();
  webgpu::g_queue.Submit(1, &commands);

  bool mapped = false;
  const auto future = buffer.MapAsync(
      wgpu::MapMode::Read, 0, bufferDescriptor.size, wgpu::CallbackMode::WaitAnyOnly,
      [&mapped](wgpu::MapAsyncStatus status, wgpu::StringView) { mapped = status == wgpu::MapAsyncStatus::Success; });
  if (webgpu::g_instance.WaitAny(future, UINT64_MAX) != wgpu::WaitStatus::Success || !mapped) {
    Log.error("Failed to map framebuffer readback buffer");
    return false;
  }
  const auto* data = static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, bufferDescriptor.size));
  auto* out = static_cast<uint8_t*>(dst);
  for (uint32_t y = 0; y < height; ++y) {
    std::memcpy(out + y * rowSize, data + size_t{y} * pitch, rowSize);
    if (bgra) {
      for (size_t x = 0; x < rowSize; x += 4) {
        std::swap(out[y * rowSize + x], out[y * rowSize + x + 2]);
      }
    }
  }
  buffer.Unmap();
  return true;
}
} // namespace aurora::gfx

bool aurora_read_framebuffer(void* dst, size_t dstSize, uint32_t* width, uint32_t* height) {
  return aurora::gfx::read_framebuffer(dst, dstSize, width, height);
}
