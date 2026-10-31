#include <gtest/gtest.h>

#include "readback_test_util.hpp"

#include "gfx/frame.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/readback/copy_layout.hpp"
#include "gfx/readback/gpu.hpp"
#include "gfx/readback/readback.hpp"
#include "gfx/recording.hpp"
#include "gfx/render_worker.hpp"
#include "gfx/resources.hpp"
#include "gfx/texture.hpp"
#include "gx/fifo.hpp"
#include "gx/readback.hpp"
#include "gx/texture.hpp"
#include "webgpu/gpu.hpp"

#include <dolphin/gx.h>

#include "internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <random>
#include <thread>

namespace {
using namespace aurora;
namespace rb = gfx::readback;

TEST(ReadbackColorLayout, PortableFallbackHandlesTailsAndUnalignedBuffers) {
  for (size_t pixels = 0; pixels <= 129; ++pixels) {
    for (const size_t srcOffset : {0u, 1u, 3u}) {
      for (const size_t dstOffset : {0u, 1u, 3u}) {
        SCOPED_TRACE(::testing::Message()
                     << "pixels=" << pixels << " srcOffset=" << srcOffset << " dstOffset=" << dstOffset);
        std::vector<uint8_t> source(srcOffset + pixels * 4);
        std::vector<uint8_t> dest(dstOffset + pixels * 4, 0xef);
        auto expected = dest;
        for (size_t i = 0; i < source.size(); ++i) {
          source[i] = static_cast<uint8_t>(i * 37 + i / 17);
        }
        for (size_t x = 0; x < pixels; ++x) {
          for (size_t channel = 0; channel < 4; ++channel) {
            const size_t srcChannel = channel == 0 ? 2 : channel == 2 ? 0 : channel;
            expected[dstOffset + x * 4 + channel] = source[srcOffset + x * 4 + srcChannel];
          }
        }

        rb::copy_bgra_to_rgba(dest.data() + dstOffset, source.data() + srcOffset, pixels);
        EXPECT_EQ(dest, expected);
        std::fill(dest.begin(), dest.end(), 0xef);
        rb::copy_color_rows(dest.data() + dstOffset, source.data() + srcOffset, pixels, 1, pixels * 4, true);
        EXPECT_EQ(dest, expected);
      }
    }
  }
}

TEST(ReadbackColorLayout, EmptyImagesDoNotAccessMemory) {
  for (const bool bgra : {false, true}) {
    rb::copy_color_rows(nullptr, nullptr, 0, 3, 0, bgra);
    rb::copy_color_rows(nullptr, nullptr, 3, 0, 256, bgra);
  }
}

TEST(ReadbackColorLayout, PacksRowsAndPreservesChannels) {
  for (const bool bgra : {false, true}) {
    for (const size_t width : {1u, 2u, 3u, 4u, 7u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 257u}) {
      for (const bool padded : {false, true}) {
        for (const size_t offset : {0u, 1u, 3u}) {
          constexpr size_t height = 3;
          const size_t rowBytes = width * 4;
          const size_t pitch = padded ? (rowBytes + 255) / 256 * 256 + 256 : rowBytes;
          SCOPED_TRACE(::testing::Message()
                       << "bgra=" << bgra << " width=" << width << " pitch=" << pitch << " offset=" << offset);
          // End allocations at the last pixel so sanitizers also catch SIMD tail overreads/writes.
          std::vector<uint8_t> source(offset + pitch * (height - 1) + rowBytes, 0xcd);
          std::vector<uint8_t> dest(offset + rowBytes * height, 0xef);
          std::vector<uint8_t> expected(dest.size(), 0xef);
          for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
              const std::array<uint8_t, 4> rgba{static_cast<uint8_t>(x + y * 7), static_cast<uint8_t>(x * 3 + 19),
                                                static_cast<uint8_t>(y * 13 + 83), static_cast<uint8_t>(x ^ y ^ 0xa5)};
              auto* pixel = source.data() + offset + y * pitch + x * 4;
              pixel[0] = rgba[bgra ? 2 : 0];
              pixel[1] = rgba[1];
              pixel[2] = rgba[bgra ? 0 : 2];
              pixel[3] = rgba[3];
              std::memcpy(expected.data() + offset + y * rowBytes + x * 4, rgba.data(), 4);
            }
          }
          const auto original = source;
          rb::copy_color_rows(dest.data() + offset, source.data() + offset, width, height, pitch, bgra);
          EXPECT_EQ(dest, expected);
          EXPECT_EQ(source, original);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// GX texture layouts
// ---------------------------------------------------------------------------------------------------------------

std::vector<uint8_t> solid_texels(uint32_t width, uint32_t height, std::array<uint8_t, 4> rgba) {
  std::vector<uint8_t> texels(size_t{width} * height * 4);
  for (size_t i = 0; i < texels.size(); i += 4) {
    std::memcpy(&texels[i], rgba.data(), 4);
  }
  return texels;
}

// Texel value encodes its coordinates: r = x, g = y.
std::vector<uint8_t> coordinate_texels(uint32_t width, uint32_t height) {
  std::vector<uint8_t> texels(size_t{width} * height * 4);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint8_t* p = &texels[(size_t{y} * width + x) * 4];
      p[0] = static_cast<uint8_t>(x);
      p[1] = static_cast<uint8_t>(y);
      p[2] = static_cast<uint8_t>(x ^ y);
      p[3] = static_cast<uint8_t>(x + y);
    }
  }
  return texels;
}

uint32_t reference_offset_8bit(uint32_t x, uint32_t y, uint32_t log2Width) {
  return (y & 3u) * 8u + ((x >> 3u) + ((y >> 2u) << (log2Width - 3u))) * 32u + (x & 7u);
}

TEST(ReadbackLayout, EightBitMatchesGameIndexing) {
  for (uint32_t log2Width = 3; log2Width <= 9; ++log2Width) {
    const uint32_t width = 1u << log2Width;
    for (uint32_t y = 0; y < 16; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        ASSERT_EQ(rb::test::texel_offset_8bit(x, y, width), reference_offset_8bit(x, y, log2Width));
      }
    }
  }
}

TEST(ReadbackLayout, ByteSourceInvertsTexelOffset) {
  const auto format = *rb::format_info(GX_CTF_A8);
  for (const uint32_t width : {8u, 20u, 64u}) {
    const uint32_t size = rb::byte_size(GX_CTF_A8, width, 12);
    std::vector<bool> seen(size);
    for (uint32_t offset = 0; offset < size; ++offset) {
      const auto source = rb::test::byte_source(format, offset, width);
      const uint32_t back = rb::test::texel_offset_8bit(source.texel.x, source.texel.y, width);
      ASSERT_EQ(back, offset);
      ASSERT_FALSE(seen[back]);
      seen[back] = true;
    }
  }
}

TEST(ReadbackLayout, SizesMatchGXGetTexBufferSize) {
  for (const uint32_t fmt :
       {GX_TF_I4,   GX_TF_I8,   GX_TF_IA4,   GX_TF_IA8, GX_TF_RGB565, GX_TF_RGB5A3, GX_TF_RGBA8, GX_CTF_R4,
        GX_CTF_RA4, GX_CTF_RA8, GX_CTF_A8,   GX_CTF_R8, GX_CTF_G8,    GX_CTF_B8,    GX_CTF_RG8,  GX_CTF_GB8,
        GX_TF_Z8,   GX_TF_Z16,  GX_TF_Z24X8, GX_CTF_Z4, GX_CTF_Z8M,   GX_CTF_Z8L,   GX_CTF_Z16L}) {
    for (const auto [w, h] : {std::pair{32u, 64u}, std::pair{20u, 6u}, std::pair{1u, 1u}, std::pair{640u, 480u}}) {
      EXPECT_EQ(rb::byte_size(fmt, w, h), GXGetTexBufferSize(w, h, fmt, GX_FALSE, 0))
          << "format " << fmt << " " << w << "x" << h;
    }
  }
  EXPECT_FALSE(rb::is_supported(GX_TF_CMPR));
  EXPECT_FALSE(rb::is_supported(GX_CTF_YUVA8));
}

TEST(ReadbackLayout, Z16StoresMiddleByteFirst) {
  // Z16 copies use the RG8 path with G and R swapped: byte 0 = Z[15:8], byte 1 = Z[23:16].
  // The copy texture holds I = Z[23:16] in RGB and A = Z[15:8] in alpha.
  const auto texels = solid_texels(4, 4, {0xAB, 0xAB, 0xAB, 0xCD});
  const auto bytes = rb::test::pack_reference(GX_TF_Z16, 4, 4, 4, 4, texels, false);
  ASSERT_EQ(bytes.size(), 32u);
  for (size_t i = 0; i < bytes.size(); i += 2) {
    EXPECT_EQ(bytes[i], 0xCD);
    EXPECT_EQ(bytes[i + 1], 0xAB);
  }
  const auto low =
      rb::test::pack_reference(GX_CTF_Z8L, 8, 4, 8, 4, solid_texels(8, 4, {0x5E, 0x5E, 0x5E, 0x5E}), false);
  EXPECT_EQ((uint32_t{bytes[1]} << 16) | (uint32_t{bytes[0]} << 8) | low[0], 0xABCD5Eu);
}

TEST(ReadbackLayout, SixteenBitTilesAreFourByFour) {
  const auto texels = coordinate_texels(8, 8);
  const auto bytes = rb::test::pack_reference(GX_CTF_RA8, 8, 8, 8, 8, texels, false);
  ASSERT_EQ(bytes.size(), 128u);
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      const size_t offset = (((y / 4) * 2 + x / 4) * 16 + (y % 4) * 4 + x % 4) * 2;
      EXPECT_EQ(bytes[offset], static_cast<uint8_t>(x + y));
      EXPECT_EQ(bytes[offset + 1], static_cast<uint8_t>(x));
    }
  }
}

TEST(ReadbackLayout, Rgba8SplitsArAndGbBlocks) {
  const auto texels = coordinate_texels(4, 4);
  const auto bytes = rb::test::pack_reference(GX_TF_RGBA8, 4, 4, 4, 4, texels, false);
  ASSERT_EQ(bytes.size(), 64u);
  for (uint32_t t = 0; t < 16; ++t) {
    const uint32_t x = t % 4;
    const uint32_t y = t / 4;
    EXPECT_EQ(bytes[t * 2], static_cast<uint8_t>(x + y)) << "A";
    EXPECT_EQ(bytes[t * 2 + 1], static_cast<uint8_t>(x)) << "R";
    EXPECT_EQ(bytes[32 + t * 2], static_cast<uint8_t>(y)) << "G";
    EXPECT_EQ(bytes[32 + t * 2 + 1], static_cast<uint8_t>(x ^ y)) << "B";
  }
}

TEST(ReadbackLayout, FourBitPacksFirstTexelHigh) {
  std::vector<uint8_t> texels(8 * 8 * 4);
  for (uint32_t i = 0; i < 64; ++i) {
    const uint8_t n = static_cast<uint8_t>((i % 16) * 17);
    std::memset(&texels[i * 4], n, 4);
  }
  const auto bytes = rb::test::pack_reference(GX_TF_I4, 8, 8, 8, 8, texels, false);
  ASSERT_EQ(bytes.size(), 32u);
  EXPECT_EQ(bytes[0], 0x01);
  EXPECT_EQ(bytes[1], 0x23);
  EXPECT_EQ(bytes[4], 0x89); // row 1 starts at texel 8
}

TEST(ReadbackLayout, ColorEncodings) {
  const auto rgb565 =
      rb::test::pack_reference(GX_TF_RGB565, 4, 4, 4, 4, solid_texels(4, 4, {0xFF, 0x80, 0x08, 0xFF}), false);
  EXPECT_EQ(rgb565[0], 0xFC); // 11111 100
  EXPECT_EQ(rgb565[1], 0x01); // 000 00001
  const auto opaque =
      rb::test::pack_reference(GX_TF_RGB5A3, 4, 4, 4, 4, solid_texels(4, 4, {0xF8, 0x08, 0x10, 0xE0}), false);
  EXPECT_EQ(opaque[0], 0xFC); // 1 11111 00
  EXPECT_EQ(opaque[1], 0x22); // 001 00010
  const auto translucent =
      rb::test::pack_reference(GX_TF_RGB5A3, 4, 4, 4, 4, solid_texels(4, 4, {0xF0, 0x80, 0x10, 0x60}), false);
  EXPECT_EQ(translucent[0], 0x3F); // 0 011 1111
  EXPECT_EQ(translucent[1], 0x81); // 1000 0001
  const auto ia4 = rb::test::pack_reference(GX_TF_IA4, 8, 4, 8, 4, solid_texels(8, 4, {0x11, 0x11, 0x11, 0xEE}), false);
  EXPECT_EQ(ia4[0], 0xE1);
}

TEST(ReadbackLayout, EdgeTilesArePaddedWithZero) {
  const auto texels = solid_texels(20, 6, {0x7F, 0x7F, 0x7F, 0x7F});
  const auto bytes = rb::test::pack_reference(GX_CTF_R8, 20, 6, 20, 6, texels, false);
  ASSERT_EQ(bytes.size(), 24u * 8u);
  const auto format = *rb::format_info(GX_CTF_R8);
  for (uint32_t offset = 0; offset < bytes.size(); ++offset) {
    const auto source = rb::test::byte_source(format, offset, 20);
    const bool inside = source.texel.x < 20 && source.texel.y < 6;
    ASSERT_EQ(bytes[offset], inside ? 0x7F : 0x00) << "offset " << offset;
  }
}

TEST(ReadbackLayout, ScaledCopiesSampleLogicalTexelCenters) {
  // A 2x copy texture: logical texel x reads source texel 2x + 1.
  const auto texels = coordinate_texels(16, 8);
  const auto bytes = rb::test::pack_reference(GX_CTF_R8, 8, 4, 16, 8, texels, false);
  for (uint32_t x = 0; x < 8; ++x) {
    EXPECT_EQ(bytes[x], 2 * x + 1);
  }
}

TEST(ReadbackLayout, Rgba6QuantizesRawColorOnly) {
  const auto texels = solid_texels(8, 4, {0x13, 0x13, 0x13, 0x13});
  EXPECT_EQ(rb::test::pack_reference(GX_CTF_A8, 8, 4, 8, 4, texels, true)[0], 0x10);
  EXPECT_EQ(rb::test::pack_reference(GX_TF_I8, 8, 4, 8, 4, texels, true)[0], 0x13);
  EXPECT_EQ(rb::test::pack_reference(GX_CTF_Z8L, 8, 4, 8, 4, texels, true)[0], 0x13);
}

// ---------------------------------------------------------------------------------------------------------------
// GPU tests
// ---------------------------------------------------------------------------------------------------------------

constexpr auto ColorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto DepthFormat = wgpu::TextureFormat::Depth32Float;
constexpr uint32_t EfbWidth = 640;
constexpr uint32_t EfbHeight = 480;

webgpu::TextureWithSampler create_target(wgpu::TextureFormat format, wgpu::TextureUsage usage, uint32_t width,
                                         uint32_t height, const char* label) {
  const wgpu::TextureDescriptor descriptor{
      .label = label,
      .usage = usage,
      .size = {width, height, 1},
      .format = format,
  };
  auto texture = webgpu::g_device.CreateTexture(&descriptor);
  auto view = texture.CreateView();
  return {
      .texture = std::move(texture),
      .view = std::move(view),
      .size = {width, height, 1},
      .format = format,
  };
}

class ReadbackGpuTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    const std::array features{wgpu::InstanceFeatureName::TimedWaitAny};
    const wgpu::InstanceDescriptor instanceDescriptor{
        .requiredFeatureCount = features.size(),
        .requiredFeatures = features.data(),
    };
    webgpu::g_instance = wgpu::CreateInstance(&instanceDescriptor);
    if (!webgpu::g_instance) {
      return;
    }
    const wgpu::RequestAdapterOptions options{.featureLevel = wgpu::FeatureLevel::Core};
    wgpu::Adapter adapter;
    const auto adapterFuture = webgpu::g_instance.RequestAdapter(
        &options, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView) {
          if (status == wgpu::RequestAdapterStatus::Success) {
            adapter = std::move(result);
          }
        });
    webgpu::g_instance.WaitAny(adapterFuture, 5'000'000'000);
    if (!adapter) {
      return;
    }
    wgpu::AdapterInfo info;
    if (adapter.GetInfo(&info) != wgpu::Status::Success || info.backendType == wgpu::BackendType::Null) {
      return;
    }
    wgpu::DeviceDescriptor deviceDescriptor{};
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView message) {
          ADD_FAILURE() << "WebGPU error " << static_cast<int>(type) << ": " << std::string_view{message};
        });
    const auto deviceFuture =
        adapter.RequestDevice(&deviceDescriptor, wgpu::CallbackMode::WaitAnyOnly,
                              [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView) {
                                if (status == wgpu::RequestDeviceStatus::Success) {
                                  webgpu::g_device = std::move(device);
                                }
                              });
    webgpu::g_instance.WaitAny(deviceFuture, 5'000'000'000);
    if (!webgpu::g_device) {
      return;
    }
    webgpu::g_queue = webgpu::g_device.GetQueue();
    webgpu::g_hasCoreFeatures = true;
    webgpu::g_efbFormat = ColorFormat;
    webgpu::g_graphicsConfig.depthFormat = DepthFormat;
    webgpu::g_graphicsConfig.msaaSamples = 1;
    webgpu::g_frameBuffer = create_target(ColorFormat,
                                          wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                                              wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
                                          EfbWidth, EfbHeight, "Test EFB");
    webgpu::g_depthBuffer =
        create_target(DepthFormat, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding, EfbWidth,
                      EfbHeight, "Test EFB Depth");
    static const std::string cachePath = ::testing::TempDir();
    g_config.cachePath = cachePath.c_str();
    g_config.userPath = cachePath.c_str();
    gfx::initialize();
    gx::fifo::init();
    GXInit(nullptr, 0);
    s_ready = true;
  }

  static void TearDownTestSuite() {
    if (s_ready) {
      gx::fifo::shutdown();
      gfx::render_worker::synchronize();
      gfx::shutdown();
    }
    webgpu::g_frameBuffer = {};
    webgpu::g_depthBuffer = {};
    webgpu::g_queue = {};
    webgpu::g_device = {};
    webgpu::g_instance = {};
    s_ready = false;
  }

  void SetUp() override {
    if (!s_ready) {
      GTEST_SKIP() << "No WebGPU device available";
    }
  }

  // Mirrors aurora::begin_frame/end_frame around `body`, then submits like the presenter does.
  static void run_frame(const std::function<void()>& body) {
    ASSERT_TRUE(gfx::begin_frame());
    gx::fifo::begin_frame();
    body();
    gx::fifo::drain();
    gx::fifo::end_frame();
    gx::texture::end_frame();
    gfx::finish();
    std::promise<void> submitted;
    gfx::end_frame([&](wgpu::CommandEncoder& encoder, std::vector<gfx::AfterSubmitCallback> callbacks) {
      const auto commands = encoder.Finish();
      webgpu::g_queue.Submit(1, &commands);
      for (auto& callback : callbacks) {
        callback();
      }
      submitted.set_value();
    });
    submitted.get_future().wait();
  }

  // Pumps device events until the stream publishes a packet at or after `sequence`.
  static bool wait_published(AuroraReadbackStream stream, uint64_t sequence = 1) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::chrono::steady_clock::now() < deadline) {
      webgpu::g_instance.ProcessEvents();
      if (const auto packet = rb::latest(stream); packet && packet->sequence >= sequence) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return false;
  }

  static uint32_t expected_z24(uint32_t z) {
    const float depth = 1.f - static_cast<float>(z) / 16777215.f;
    return std::min(static_cast<uint32_t>(std::clamp(1.f - depth, 0.f, 1.f) * 16777215.f + 0.5f), 0xFFFFFFu);
  }

  static inline bool s_ready = false;
};

TEST_F(ReadbackGpuTest, CaptureObservesItsCommandStreamPosition) {
  // A stream keeps only its newest capture, so captures before and after "draw B" use separate streams.
  const auto before = AuroraReadbackCreateStream();
  const auto after = AuroraReadbackCreateStream();
  const auto copies = AuroraReadbackCreateStream();
  constexpr GXColor ColorA{0x13, 0x57, 0x9B, 0x41};
  constexpr GXColor ColorB{0xC8, 0x22, 0x7E, 0x80};
  constexpr u32 DepthA = 0xFFFFFF;
  constexpr u32 DepthB = 0x3456BC;
  alignas(32) static u8 dummy[EfbWidth * EfbHeight * 4];
  alignas(32) static u8 ids[32 * 64];
  alignas(32) static u8 high[32 * 64 * 2];
  alignas(32) static u8 low[32 * 64];
  const AuroraReadbackPeekRequest points[] = {{0, AURORA_READBACK_PEEK_Z, 100, 100},
                                              {1, AURORA_READBACK_PEEK_ARGB, 100, 100}};

  // The first EFB pass of a frame clears to the copy clear values processed before the frame begins.
  run_frame([&] { GXSetCopyClear(ColorA, DepthA); });

  run_frame([&] {
    GXSetPixelFmt(GX_PF_RGBA6_Z24, GX_ZC_LINEAR);
    GXPokeAlphaRead(GX_READ_NONE);
    EXPECT_TRUE(AuroraReadbackCapturePeeks(before, points, 2));

    // "Draw B": a clearing copy rewrites every EFB pixel after the first capture.
    GXSetCopyClear(ColorB, DepthB);
    GXSetTexCopySrc(0, 0, EfbWidth, EfbHeight);
    GXSetTexCopyDst(EfbWidth, EfbHeight, GX_TF_RGBA8, GX_FALSE);
    GXCopyTex(dummy, GX_TRUE);

    EXPECT_TRUE(AuroraReadbackCapturePeeks(after, points, 2));

    GXSetTexCopySrc(304, 192, 32, 64);
    GXSetTexCopyDst(32, 64, GX_CTF_A8, GX_FALSE);
    GXCopyTex(ids, GX_FALSE);
    GXSetTexCopyDst(32, 64, GX_TF_Z16, GX_FALSE);
    GXCopyTex(high, GX_FALSE);
    GXSetTexCopyDst(32, 64, GX_CTF_Z8L, GX_FALSE);
    GXCopyTex(low, GX_FALSE);
    const AuroraReadbackCopySrc sources[] = {{1, ids}, {2, high}, {3, low}};
    EXPECT_TRUE(AuroraReadbackCaptureCopies(copies, sources, 3));
  });
  ASSERT_TRUE(wait_published(before));
  ASSERT_TRUE(wait_published(after));
  ASSERT_TRUE(wait_published(copies));

  constexpr auto q = [](u32 c) { return (c & 0xFC) | (c >> 6); };
  u32 z = 0;
  u32 argb = 0;
  AuroraReadbackDest results[] = {{0, &z, sizeof(u32)}, {1, &argb, sizeof(u32)}};
  ASSERT_TRUE(AuroraReadbackGet(before, results, 2));
  ASSERT_TRUE(results[0].found);
  ASSERT_TRUE(results[1].found);
  EXPECT_EQ(z, DepthA);
  EXPECT_EQ(argb, (q(ColorA.a) << 24) | (q(ColorA.r) << 16) | (q(ColorA.g) << 8) | q(ColorA.b));
  ASSERT_TRUE(AuroraReadbackGet(after, results, 2));
  ASSERT_TRUE(results[0].found);
  ASSERT_TRUE(results[1].found);
  EXPECT_NEAR(static_cast<double>(z), static_cast<double>(expected_z24(DepthB)), 1.0);
  EXPECT_EQ(argb, (q(ColorB.a) << 24) | (q(ColorB.r) << 16) | (q(ColorB.g) << 8) | q(ColorB.b));

  std::array<u8, sizeof(ids)> outIds{};
  std::array<u8, sizeof(high)> outHigh{};
  std::array<u8, sizeof(low)> outLow{};
  AuroraReadbackDest outputs[] = {
      {1, outIds.data(), sizeof(ids)},
      {2, outHigh.data(), sizeof(high)},
      {3, outLow.data(), sizeof(low)},
  };
  ASSERT_TRUE(AuroraReadbackGet(copies, outputs, 3));
  for (const auto& output : outputs) {
    ASSERT_TRUE(output.found) << "key " << output.key;
  }
  for (const u8 id : outIds) {
    ASSERT_EQ(id, q(ColorB.a));
  }
  for (size_t i = 0; i < outLow.size(); ++i) {
    const u32 z24 = (u32{outHigh[i * 2 + 1]} << 16) | (u32{outHigh[i * 2]} << 8) | outLow[i];
    ASSERT_NEAR(static_cast<double>(z24), static_cast<double>(expected_z24(DepthB)), 1.0) << "texel byte " << i;
  }
  AuroraReadbackDestroyStream(before);
  AuroraReadbackDestroyStream(after);
  AuroraReadbackDestroyStream(copies);
}

// Runs one job directly on the GPU and waits for its publication.
std::shared_ptr<const rb::Packet> run_job(const std::shared_ptr<rb::Job>& job, AuroraReadbackStream stream) {
  gfx::detail::FramePacket frame;
  const wgpu::CommandEncoderDescriptor descriptor{.label = "Readback test"};
  auto encoder = webgpu::g_device.CreateCommandEncoder(&descriptor);
  rb::gpu::encode_job(encoder, frame, job);
  const auto commands = encoder.Finish();
  webgpu::g_queue.Submit(1, &commands);
  for (auto& callback : frame.afterSubmitCallbacks) {
    callback();
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (std::chrono::steady_clock::now() < deadline) {
    webgpu::g_instance.ProcessEvents();
    if (auto packet = rb::latest(stream); packet && packet->sequence >= job->packet->sequence) {
      return packet;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return nullptr;
}

uint32_t write_uniform(uint32_t& offset, const void* data, size_t size) {
  const uint32_t at = offset;
  webgpu::g_queue.WriteBuffer(gfx::detail::resources().uniformBuffer, at, data, size);
  offset += 256;
  return at;
}

TEST_F(ReadbackGpuTest, PackShaderMatchesReferenceForEveryFormat) {
  const auto stream = AuroraReadbackCreateStream();
  std::mt19937 rng{1234};
  constexpr uint32_t Width = 20; // partial tiles in both directions for every family
  constexpr uint32_t Height = 12;
  constexpr uint32_t SrcWidth = 40;
  constexpr uint32_t SrcHeight = 24;
  const std::array formats{GX_TF_I4,    GX_TF_I8,  GX_TF_IA4,  GX_TF_IA8,  GX_TF_RGB565, GX_TF_RGB5A3,
                           GX_TF_RGBA8, GX_CTF_R4, GX_CTF_RA4, GX_CTF_RA8, GX_CTF_A8,    GX_CTF_R8,
                           GX_CTF_G8,   GX_CTF_B8, GX_CTF_RG8, GX_CTF_GB8, GX_TF_Z8,     GX_TF_Z16,
                           GX_TF_Z24X8, GX_CTF_Z4, GX_CTF_Z8M, GX_CTF_Z8L, GX_CTF_Z16L};
  std::vector<rb::Output> outputs;
  std::vector<std::vector<uint8_t>> sources;
  uint32_t words = 0;
  for (size_t i = 0; i < formats.size(); ++i) {
    std::vector<uint8_t> texels(size_t{SrcWidth} * SrcHeight * 4);
    for (auto& b : texels) {
      b = static_cast<uint8_t>(rng());
    }
    const uint32_t size = rb::byte_size(formats[i], Width, Height);
    outputs.push_back({.key = i, .payloadOffset = words * 4, .payloadSize = size});
    sources.push_back(std::move(texels));
    words += size / 4;
  }
  auto job = rb::create_job(stream, 1, rb::JobKind::GxCopy, outputs, words * 4);
  uint32_t uniformOffset = 0;
  for (size_t i = 0; i < formats.size(); ++i) {
    auto texture = create_target(ColorFormat, wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
                                 SrcWidth, SrcHeight, "Copy source");
    const wgpu::TexelCopyTextureInfo destination{.texture = texture.texture};
    const wgpu::TexelCopyBufferLayout dataLayout{.bytesPerRow = SrcWidth * 4, .rowsPerImage = SrcHeight};
    const wgpu::Extent3D extent{SrcWidth, SrcHeight, 1};
    webgpu::g_queue.WriteTexture(&destination, sources[i].data(), sources[i].size(), &dataLayout, &extent);
    const auto format = *rb::format_info(formats[i]);
    const bool rgba6 = i % 2 == 0;
    const rb::CopyPackParams params{
        .family = static_cast<uint32_t>(format.family),
        .encoding = static_cast<uint32_t>(format.encoding),
        .tileWidth = format.tileWidth,
        .tileHeight = format.tileHeight,
        .tileBytes = format.tileBytes,
        .tilesX = rb::tiles_x(format, Width),
        .width = Width,
        .height = Height,
        .srcWidth = SrcWidth,
        .srcHeight = SrcHeight,
        .wordCount = outputs[i].payloadSize / 4,
        .outWord = outputs[i].payloadOffset / 4,
        .quantizeRgba6 = rgba6 && format.rawColor ? 1u : 0u,
    };
    const auto offset = write_uniform(uniformOffset, &params, sizeof(params));
    job->copies.push_back({
        .texture = std::make_shared<gfx::TextureRef>(texture.texture, texture.view, wgpu::TextureView{}, extent,
                                                     ColorFormat, 1, formats[i]),
        .params = {offset, sizeof(params)},
        .wordCount = params.wordCount,
    });
  }
  const auto packet = run_job(job, stream);
  ASSERT_TRUE(packet);
  for (size_t i = 0; i < formats.size(); ++i) {
    const auto* output = packet->find(i);
    ASSERT_NE(output, nullptr);
    const auto expected =
        rb::test::pack_reference(formats[i], Width, Height, SrcWidth, SrcHeight, sources[i], i % 2 == 0);
    const auto actual = packet->payload_of(*output);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t b = 0; b < expected.size(); ++b) {
      ASSERT_EQ(actual[b], expected[b]) << "format " << formats[i] << " byte " << b;
    }
  }
  AuroraReadbackDestroyStream(stream);
}

TEST_F(ReadbackGpuTest, GatherShaderMatchesReference) {
  const auto stream = AuroraReadbackCreateStream();
  constexpr uint32_t TargetWidth = 1280; // 2x upscaled render target
  constexpr uint32_t TargetHeight = 960;
  auto color = create_target(ColorFormat, wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst, TargetWidth,
                             TargetHeight, "Gather color");
  std::vector<uint8_t> texels(size_t{TargetWidth} * TargetHeight * 4);
  std::mt19937 rng{99};
  for (auto& b : texels) {
    b = static_cast<uint8_t>(rng());
  }
  const wgpu::TexelCopyTextureInfo destination{.texture = color.texture};
  const wgpu::TexelCopyBufferLayout dataLayout{.bytesPerRow = TargetWidth * 4, .rowsPerImage = TargetHeight};
  const wgpu::Extent3D extent{TargetWidth, TargetHeight, 1};
  webgpu::g_queue.WriteTexture(&destination, texels.data(), texels.size(), &dataLayout, &extent);

  // Depth encodes the pixel position; exact Z24 values are compared with one LSB of conversion tolerance.
  auto depth = create_target(DepthFormat, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
                             TargetWidth, TargetHeight, "Gather depth");
  {
    constexpr std::string_view Source = R"(
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));
    return vec4f(positions[vi], 0.0, 1.0);
}
struct Out { @builtin(frag_depth) depth: f32 };
@fragment fn fs_main(@builtin(position) pos: vec4f) -> Out {
    let p = vec2u(pos.xy);
    return Out(f32(p.y * 1280u + p.x) / f32(1280u * 960u));
}
)";
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = Source.data()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    const auto module = webgpu::g_device.CreateShaderModule(&moduleDescriptor);
    const wgpu::DepthStencilState depthState{
        .format = DepthFormat,
        .depthWriteEnabled = true,
        .depthCompare = wgpu::CompareFunction::Always,
    };
    const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs_main"};
    const wgpu::RenderPipelineDescriptor pipelineDescriptor{
        .vertex = {.module = module, .entryPoint = "vs_main"},
        .depthStencil = &depthState,
        .fragment = &fragment,
    };
    const auto pipeline = webgpu::g_device.CreateRenderPipeline(&pipelineDescriptor);
    const wgpu::RenderPassDepthStencilAttachment attachment{
        .view = depth.view,
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 0.f,
    };
    const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &attachment};
    auto encoder = webgpu::g_device.CreateCommandEncoder();
    auto pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(pipeline);
    pass.Draw(3);
    pass.End();
    const auto commands = encoder.Finish();
    webgpu::g_queue.Submit(1, &commands);
  }
  const auto depth_at = [](uint32_t x, uint32_t y) {
    const float d = static_cast<float>(y * TargetWidth + x) / static_cast<float>(TargetWidth * TargetHeight);
    return std::min(static_cast<uint32_t>(std::clamp(1.f - d, 0.f, 1.f) * 16777215.f + 0.5f), 0xFFFFFFu);
  };
  const rb::test::EfbModel model{
      .depth = depth_at,
      .color =
          [&](uint32_t x, uint32_t y) {
            const uint8_t* p = &texels[(size_t{y} * TargetWidth + x) * 4];
            return std::array<uint8_t, 4>{p[0], p[1], p[2], p[3]};
          },
  };

  std::vector<rb::EfbGatherJob> jobs{
      {.x = 0, .y = 0, .width = 1, .height = 1, .outWord = 0, .color = 0, .alphaMode = GX_READ_NONE},
      {.x = 639, .y = 479, .width = 1, .height = 1, .outWord = 1, .color = 1, .alphaMode = GX_READ_NONE},
      {.x = 10, .y = 20, .width = 32, .height = 64, .outWord = 2, .color = 1, .alphaMode = GX_READ_00},
      {.x = 300, .y = 200, .width = 32, .height = 64, .outWord = 2 + 2048, .color = 0, .alphaMode = GX_READ_FF},
      {.x = 5, .y = 7, .width = 1, .height = 1, .outWord = 2 + 4096, .color = 1, .alphaMode = GX_READ_FF},
  };
  const uint32_t words = 2 + 4096 + 1;
  uint64_t sequence = 0;
  for (const uint32_t pixelFormat : {GX_PF_RGBA6_Z24, GX_PF_RGB8_Z24, GX_PF_RGB565_Z16}) {
    ++sequence;
    for (auto& job : jobs) {
      job.pixelFormat = pixelFormat;
    }
    const rb::EfbGatherParams params{
        .jobCount = static_cast<uint32_t>(jobs.size()),
        .totalWords = words,
        .scaleNumX = TargetWidth,
        .scaleDenX = EfbWidth,
        .scaleNumY = TargetHeight,
        .scaleDenY = EfbHeight,
        .targetWidth = TargetWidth,
        .targetHeight = TargetHeight,
    };
    std::vector<rb::Output> outputs{{.key = 0, .payloadSize = words * 4}};
    auto job = rb::create_job(stream, sequence, rb::JobKind::EfbGather, outputs, words * 4);
    uint32_t uniformOffset = 0;
    job->gatherParams = {write_uniform(uniformOffset, &params, sizeof(params)), sizeof(params)};
    webgpu::g_queue.WriteBuffer(gfx::detail::resources().storageBuffer, 0, jobs.data(),
                                jobs.size() * sizeof(rb::EfbGatherJob));
    job->gatherJobs = {0, static_cast<uint32_t>(jobs.size() * sizeof(rb::EfbGatherJob))};
    job->gatherWords = words;
    job->depthView = depth.view;
    job->colorView = color.view;
    const auto packet = run_job(job, stream);
    ASSERT_TRUE(packet);
    ASSERT_EQ(packet->sequence, sequence);
    const auto expected = rb::test::gather_reference(params, jobs, model);
    for (uint32_t w = 0; w < words; ++w) {
      uint32_t actual;
      uint32_t wanted;
      std::memcpy(&actual, &packet->payload[w * 4], 4);
      std::memcpy(&wanted, &expected[w * 4], 4);
      const bool isDepth = w == 0 || (w >= 2 + 2048 && w < 2 + 4096);
      if (isDepth) {
        ASSERT_LE(std::max(actual, wanted) - std::min(actual, wanted), 2u) << "word " << w;
      } else {
        ASSERT_EQ(actual, wanted) << "word " << w << " pixel format " << pixelFormat;
      }
    }
  }
  AuroraReadbackDestroyStream(stream);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Raw hardware streams through the address resolver: EFB copies to guest memory, copy writeback, display copies
// without clear, Z textures and logic ops
// ---------------------------------------------------------------------------------------------------------------

namespace {
constexpr u32 GuestBase = 0x00800000;
alignas(32) std::array<u8, 1 << 20> sGuest{};

const void* resolve_guest(u32 address, u32* size, void*) {
  const u32 phys = address & 0x3FFFFFFF;
  if (phys < GuestBase || phys - GuestBase >= sGuest.size()) {
    return nullptr;
  }
  *size = static_cast<u32>(sGuest.size()) - (phys - GuestBase);
  return sGuest.data() + (phys - GuestBase);
}

struct WrittenCopy {
  AuroraCopyWritebackInfo info;
  std::vector<u8> data;
};
std::vector<WrittenCopy> sWritten;

void record_writeback(const AuroraCopyWritebackInfo* copy, void*) {
  const auto* bytes = static_cast<const u8*>(copy->data);
  sWritten.push_back({*copy, std::vector<u8>(bytes, bytes + copy->size)});
  std::memcpy(sGuest.data() + (copy->address - GuestBase), copy->data, copy->size);
}

// A BP register write in the raw stream
void bp(u32 value) {
  gx::fifo::write_u8(GX_LOAD_BP_REG);
  gx::fifo::write_u32(value);
}

// EFB copy registers as the SDK writes them (GXSetTexCopySrc/Dst, GXCopyTex): RGBA8 unless `format` (the 4-bit PE
// copy format) is given, clamped top and bottom.
void guest_copy(u32 left, u32 top, u32 width, u32 height, u32 address, bool half, bool clear, u32 format = 6,
                bool intensity = false) {
  bp(0x49000000 | left | (top << 10));
  bp(0x4A000000 | (width - 1) | ((height - 1) << 10));
  bp(0x4B000000 | (address >> 5));
  bp(0x4D000000 | 0);
  bp(0x52000000 | 3 | ((format >> 3) & 1) << 3 | (format & 7) << 4 | (half ? 1u : 0u) << 9 |
     (clear ? 1u : 0u) << 11 | (intensity ? 3u : 2u) << 15);
}

void display_copy(bool clear) {
  bp(0x49000000);
  bp(0x4A000000 | (EfbWidth - 1) | ((EfbHeight - 1) << 10));
  bp(0x4B000000 | ((GuestBase + 0x80000) >> 5));
  bp(0x52000000 | 3 | (clear ? 1u : 0u) << 11 | 1u << 14);
}

// Texture map 0 from guest memory: nearest, clamped, no mipmaps
void guest_texture(u32 address, u32 width, u32 height, u32 format) {
  bp(0x80000000);
  bp(0x84000000);
  bp(0x88000000 | (width - 1) | (height - 1) << 10 | format << 20);
  bp(0x94000000 | (address >> 5));
}

// Screen-space quads (EFB pixels), one TEV stage: texture map 0 (REPLACE) or the colour/alpha of TEV register 0
void setup_2d(bool textured) {
  GXSetViewport(0, 0, EfbWidth, EfbHeight, 0, 1);
  GXSetScissor(0, 0, EfbWidth, EfbHeight);
  f32 proj[4][4] = {};
  proj[0][0] = 2.f / EfbWidth;
  proj[0][3] = -1.f;
  proj[1][1] = -2.f / EfbHeight;
  proj[1][3] = 1.f;
  proj[2][2] = -1.f;
  proj[2][3] = -1.f;
  proj[3][3] = 1.f;
  GXSetProjection(proj, GX_ORTHOGRAPHIC);
  f32 ident[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  GXLoadPosMtxImm(ident, GX_PNMTX0);
  GXSetCurrentMtx(GX_PNMTX0);
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);
  GXSetNumChans(0);
  GXSetNumTexGens(1);
  GXSetTexCoordGen(GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY);
  GXSetNumTevStages(1);
  GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, textured ? GX_TEXMAP0 : GX_TEXMAP_NULL, GX_COLOR_NULL);
  if (textured) {
    GXSetTevOp(GX_TEVSTAGE0, GX_REPLACE);
  } else {
    GXSetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_C0);
    GXSetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_A0);
    GXSetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
  }
  GXSetAlphaCompare(GX_ALWAYS, 0, GX_AOP_AND, GX_ALWAYS, 0);
  GXSetZMode(GX_FALSE, GX_ALWAYS, GX_FALSE);
  GXSetCullMode(GX_CULL_NONE);
  GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
  GXSetColorUpdate(GX_TRUE);
  GXSetAlphaUpdate(GX_TRUE);
}

void quad(f32 x0, f32 y0, f32 x1, f32 y1) {
  GXBegin(GX_QUADS, GX_VTXFMT0, 4);
  GXPosition3f32(x0, y0, -0.5f);
  GXTexCoord2f32(0, 0);
  GXPosition3f32(x1, y0, -0.5f);
  GXTexCoord2f32(1, 0);
  GXPosition3f32(x1, y1, -0.5f);
  GXTexCoord2f32(1, 1);
  GXPosition3f32(x0, y1, -0.5f);
  GXTexCoord2f32(0, 1);
  GXEnd();
}

// Fills an EFB rectangle with a colour (a clearing copy to a scratch guest address)
void fill(u32 x, u32 y, u32 w, u32 h, GXColor color) {
  GXSetCopyClear(color, 0xFFFFFF);
  guest_copy(x, y, w, h, GuestBase + 0xC0000, false, true);
}

// Texel (x, y) of an RGBA8 texture in the hardware layout: 4x4 tiles, AR then GB
std::array<u8, 4> rgba8_texel(const std::vector<u8>& data, u32 width, u32 x, u32 y) {
  const u32 tile = (y / 4) * ((width + 3) / 4) + x / 4;
  const u32 i = (y % 4) * 4 + x % 4;
  const u8* t = data.data() + tile * 64;
  return {t[i * 2 + 1], t[32 + i * 2], t[32 + i * 2 + 1], t[i * 2]};
}

class GuestCopyGpuTest : public ReadbackGpuTest {
protected:
  void SetUp() override {
    ReadbackGpuTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    sGuest.fill(0);
    sWritten.clear();
    AuroraSetAddressResolver(resolve_guest, nullptr);
    AuroraSetCopyWriteback(record_writeback, nullptr, false);
    AuroraSetDrawsBlocking(true); // no draw is skipped while its pipeline compiles
  }
  void TearDown() override {
    AuroraSetCopyWriteback(nullptr, nullptr, false);
    AuroraSetAddressResolver(nullptr, nullptr);
  }

  // ARGB (or Z) at EFB pixels, as of the end of `body`'s frame
  static std::vector<u32> peek_after(const std::function<void()>& body, AuroraReadbackPeekKind kind,
                                     const std::vector<std::pair<u16, u16>>& points) {
    const auto stream = AuroraReadbackCreateStream();
    run_frame([&] {
      body();
      std::vector<AuroraReadbackPeekRequest> requests;
      for (u32 i = 0; i < points.size(); ++i) {
        requests.push_back({i, kind, points[i].first, points[i].second});
      }
      EXPECT_TRUE(AuroraReadbackCapturePeeks(stream, requests.data(), static_cast<u32>(requests.size())));
    });
    std::vector<u32> values(points.size(), 0xDEADBEEF);
    EXPECT_TRUE(wait_published(stream));
    std::vector<AuroraReadbackDest> dests;
    for (u32 i = 0; i < points.size(); ++i) {
      dests.push_back({i, &values[i], sizeof(u32)});
    }
    EXPECT_TRUE(AuroraReadbackGet(stream, dests.data(), static_cast<u32>(dests.size())));
    AuroraReadbackDestroyStream(stream);
    return values;
  }
};

constexpr u32 argb(GXColor c) { return u32{c.a} << 24 | u32{c.r} << 16 | u32{c.g} << 8 | c.b; }
} // namespace

TEST_F(GuestCopyGpuTest, HalfSizeCopyIsBoxFilteredAndWrittenBackInHardwareFormat) {
  // 16x16 at (32, 32): A everywhere, B in odd columns, then C in odd rows: each 2x2 block is A, B, C, C
  constexpr GXColor A{200, 0, 40, 255}, B{0, 100, 40, 255}, C{40, 60, 0, 255};
  run_frame([] {
    GXSetPixelFmt(GX_PF_RGB8_Z24, GX_ZC_LINEAR);
    fill(32, 32, 16, 16, A);
    for (u32 x = 1; x < 16; x += 2) {
      fill(32 + x, 32, 1, 16, B);
    }
    for (u32 y = 1; y < 16; y += 2) {
      fill(32, 32 + y, 16, 1, C);
    }
    guest_copy(32, 32, 16, 16, GuestBase, true, false);
  });
  EXPECT_EQ(AuroraFlushCopyWriteback(), 1u + 1u + 8u + 8u); // the fills are copies too
  const auto& copy = sWritten.back();
  EXPECT_EQ(copy.info.address, GuestBase);
  EXPECT_EQ(copy.info.format, static_cast<u32>(GX_TF_RGBA8));
  EXPECT_EQ(copy.info.width, 8);
  EXPECT_EQ(copy.info.height, 8);
  EXPECT_TRUE(copy.info.half);
  ASSERT_EQ(copy.data.size(), 8u * 8u * 4u);
  for (u32 y = 0; y < 8; ++y) {
    for (u32 x = 0; x < 8; ++x) {
      const auto t = rgba8_texel(copy.data, 8, x, y);
      EXPECT_NEAR(t[0], (A.r + B.r + 2 * C.r) / 4.0, 1.0) << x << "," << y;
      EXPECT_NEAR(t[1], (A.g + B.g + 2 * C.g) / 4.0, 1.0) << x << "," << y;
      EXPECT_NEAR(t[2], (A.b + B.b + 2 * C.b) / 4.0, 1.0) << x << "," << y;
      EXPECT_EQ(t[3], 255) << x << "," << y;
    }
  }
  // The writeback is what reached guest memory
  EXPECT_EQ(std::memcmp(sGuest.data(), copy.data.data(), copy.data.size()), 0);
}

TEST_F(GuestCopyGpuTest, VerticalFilterBlendsTheRowsAboveAndBelow) {
  // Rows alternate black and white; the SDK's copy filter registers with 16/32/16 weights
  constexpr GXColor Black{0, 0, 0, 255}, White{255, 255, 255, 255};
  run_frame([] {
    GXSetPixelFmt(GX_PF_RGB8_Z24, GX_ZC_LINEAR);
    fill(0, 64, 8, 8, Black);
    for (u32 y = 1; y < 8; y += 2) {
      fill(0, 64 + y, 8, 1, White);
    }
    bp(0x53000000 | 8 | 8 << 6 | 10 << 12 | 11 << 18);
    bp(0x54000000 | 11 | 8 << 6 | 8 << 12);
    guest_copy(0, 64, 8, 8, GuestBase, false, false);
    bp(0x53000000 | 21 << 12 | 22 << 18);
    bp(0x54000000 | 21);
  });
  AuroraFlushCopyWriteback();
  const auto& copy = sWritten.back();
  EXPECT_TRUE(copy.info.vfilter);
  for (u32 y = 1; y < 7; ++y) {
    // above + below weigh 32/64, the row 32/64: black rows get 0.5, white rows 0.5 as well
    const auto t = rgba8_texel(copy.data, 8, 3, y);
    EXPECT_NEAR(t[0], 127.5, 1.5) << "row " << y;
  }
  // Top row clamped: the row above is the row itself (black): (16 + 32) * 0 + 16 * 255 / 64
  EXPECT_NEAR(rgba8_texel(copy.data, 8, 3, 0)[0], 255.0 * 16 / 64, 1.5);
}

TEST_F(GuestCopyGpuTest, LaterTextureLoadSamplesTheCopyUntilMemoryChanges) {
  constexpr GXColor Red{255, 0, 0, 255}, Blue{0, 0, 255, 255};
  const std::vector<std::pair<u16, u16>> at{{100, 100}};
  // Frame 1: copy an 8x8 red block to guest memory; the texture load samples the GPU copy
  auto v = peek_after(
      [] {
        GXSetPixelFmt(GX_PF_RGB8_Z24, GX_ZC_LINEAR);
        fill(0, 0, 8, 8, Red);
        guest_copy(0, 0, 8, 8, GuestBase, false, false);
        setup_2d(true);
        guest_texture(GuestBase, 8, 8, GX_TF_RGBA8);
        quad(96, 96, 104, 104);
      },
      AURORA_READBACK_PEEK_ARGB, at);
  EXPECT_EQ(v[0] & 0xFFFFFF, argb(Red) & 0xFFFFFF);
  // The CPU writes a blue texture there and invalidates the texture cache: memory wins
  AuroraFlushCopyWriteback();
  for (u32 i = 0; i < 4; ++i) { // 8x8 RGBA8: 4 tiles of AR then GB
    for (u32 t = 0; t < 16; ++t) {
      u8* tile = sGuest.data() + i * 64;
      tile[t * 2] = 255;
      tile[t * 2 + 1] = Blue.r;
      tile[32 + t * 2] = Blue.g;
      tile[32 + t * 2 + 1] = Blue.b;
    }
  }
  v = peek_after(
      [] {
        bp(0x66001000); // GXInvalidateTexAll
        setup_2d(true);
        guest_texture(GuestBase, 8, 8, GX_TF_RGBA8);
        quad(96, 96, 104, 104);
      },
      AURORA_READBACK_PEEK_ARGB, at);
  EXPECT_EQ(v[0] & 0xFFFFFF, argb(Blue) & 0xFFFFFF);
}

TEST_F(GuestCopyGpuTest, DisplayCopyWithoutClearKeepsTheEfb) {
  constexpr GXColor A{10, 200, 30, 255}, B{90, 20, 160, 255};
  const std::vector<std::pair<u16, u16>> at{{320, 240}};
  run_frame([] { GXSetCopyClear(A, 0xFFFFFF); });
  // A frame that starts cleared to A and ends with a display copy without clear; the next frame keeps it
  auto v = peek_after([] { GXSetCopyClear(B, 0xFFFFFF); display_copy(false); }, AURORA_READBACK_PEEK_ARGB, at);
  EXPECT_EQ(v[0] & 0xFFFFFF, argb(A) & 0xFFFFFF);
  v = peek_after([] { display_copy(true); }, AURORA_READBACK_PEEK_ARGB, at);
  EXPECT_EQ(v[0] & 0xFFFFFF, argb(A) & 0xFFFFFF); // still A: this frame was not cleared either
  v = peek_after([] {}, AURORA_READBACK_PEEK_ARGB, at);
  EXPECT_EQ(v[0] & 0xFFFFFF, argb(B) & 0xFFFFFF); // the clearing display copy: the next frame starts at B
}

TEST_F(GuestCopyGpuTest, ZTextureReplacesAndAddsDepth) {
  // An 8x8 I8 texture of 0x40, read as an 8-bit Z texel, plus a bias
  std::fill_n(sGuest.begin() + 0x1000, 64, u8{0x40});
  const std::vector<std::pair<u16, u16>> at{{20, 20}, {60, 20}};
  const auto v = peek_after(
      [] {
        GXSetPixelFmt(GX_PF_RGB8_Z24, GX_ZC_LINEAR);
        setup_2d(true);
        guest_texture(GuestBase + 0x1000, 8, 8, GX_TF_I8);
        GXSetZMode(GX_TRUE, GX_ALWAYS, GX_TRUE);
        GXSetZTexture(GX_ZT_REPLACE, GX_TF_Z8, 0x123400);
        quad(16, 16, 24, 24);
        GXSetZTexture(GX_ZT_DISABLE, GX_TF_Z8, 0);
        quad(56, 16, 64, 24);
      },
      AURORA_READBACK_PEEK_Z, at);
  EXPECT_NEAR(static_cast<double>(v[0]), 0x123440, 1.0);
  EXPECT_GT(std::abs(static_cast<double>(v[1]) - 0x123440), 1000.0); // the quad's own depth
  // ADD: the quad's depth plus texel and bias
  const auto w = peek_after(
      [] {
        setup_2d(true);
        guest_texture(GuestBase + 0x1000, 8, 8, GX_TF_I8);
        GXSetZMode(GX_TRUE, GX_ALWAYS, GX_TRUE);
        GXSetZTexture(GX_ZT_ADD, GX_TF_Z8, 0x100);
        quad(56, 16, 64, 24);
        GXSetZTexture(GX_ZT_DISABLE, GX_TF_Z8, 0);
      },
      AURORA_READBACK_PEEK_Z, {{60, 20}});
  EXPECT_NEAR(static_cast<double>(w[0]), static_cast<double>(v[1]) + 0x140, 2.0);
}

TEST_F(GuestCopyGpuTest, LogicOpsMatchBitwiseOpsOnBinaryChannels) {
  // Per channel (R, G, B): src/dst = 1/1, 0/1, 1/0 in the first case and 0/0, 1/1, 0/1 in the second
  struct Case {
    GXColor src, dst;
  };
  constexpr Case Cases[] = {{{255, 0, 255, 255}, {255, 255, 0, 255}}, {{0, 255, 0, 255}, {0, 255, 255, 255}}};
  const auto op_value = [](GXLogicOp op, u8 s, u8 d) -> u8 {
    switch (op) {
    case GX_LO_CLEAR: return 0;
    case GX_LO_AND: return s & d;
    case GX_LO_REVAND: return s & ~d;
    case GX_LO_COPY: return s;
    case GX_LO_INVAND: return ~s & d;
    case GX_LO_NOOP: return d;
    case GX_LO_XOR: return s ^ d;
    case GX_LO_OR: return s | d;
    case GX_LO_NOR: return ~(s | d);
    case GX_LO_EQUIV: return ~(s ^ d);
    case GX_LO_INV: return ~d;
    case GX_LO_REVOR: return s | ~d;
    case GX_LO_INVCOPY: return ~s;
    case GX_LO_INVOR: return ~s | d;
    case GX_LO_NAND: return ~(s & d);
    case GX_LO_SET: return 0xFF;
    }
    return 0;
  };
  for (u32 o = 0; o < 16; ++o) {
    const auto op = static_cast<GXLogicOp>(o);
    if (op == GX_LO_REVOR || op == GX_LO_NAND) {
      continue; // no blend equivalent (drawn as COPY and INVCOPY)
    }
    for (const auto& c : Cases) {
      const auto v = peek_after(
          [&] {
            GXSetPixelFmt(GX_PF_RGB8_Z24, GX_ZC_LINEAR);
            fill(200, 200, 16, 16, c.dst);
            setup_2d(false);
            GXSetTevColor(GX_TEVREG0, c.src);
            GXSetBlendMode(GX_BM_LOGIC, GX_BL_ONE, GX_BL_ZERO, op);
            quad(200, 200, 216, 216);
          },
          AURORA_READBACK_PEEK_ARGB, {{208, 208}});
      const u32 want = u32{op_value(op, c.src.r, c.dst.r)} << 16 | u32{op_value(op, c.src.g, c.dst.g)} << 8 |
                       op_value(op, c.src.b, c.dst.b);
      EXPECT_EQ(v[0] & 0xFFFFFF, want) << "logic op " << o;
    }
  }
  // OR onto zero keeps any value (the depth-of-field fog pass)
  const auto v = peek_after(
      [] {
        fill(200, 200, 16, 16, {0, 0, 0, 255});
        setup_2d(false);
        GXSetTevColor(GX_TEVREG0, {0x00, 0x12, 0x3F, 0xFF});
        GXSetBlendMode(GX_BM_LOGIC, GX_BL_ONE, GX_BL_ONE, GX_LO_OR);
        quad(200, 200, 216, 216);
      },
      AURORA_READBACK_PEEK_ARGB, {{208, 208}});
  EXPECT_EQ(v[0] & 0xFFFFFF, 0x00123Fu);
}

TEST_F(GuestCopyGpuTest, AlphaOnlyClearingCopyThenDraw) {
  // A copy that clears only the alpha (colour update off), then a draw: both take effect
  constexpr GXColor Red{255, 0, 0, 200}, Blue{0, 0, 255, 255};
  const auto v = peek_after(
      [] {
        GXSetPixelFmt(GX_PF_RGBA6_Z24, GX_ZC_LINEAR);
        GXPokeAlphaRead(GX_READ_NONE);
        fill(0, 0, 64, 64, Red);
        GXSetColorUpdate(GX_FALSE);
        GXSetAlphaUpdate(GX_TRUE);
        GXSetCopyClear({0, 0, 0, 0}, 0xFFFFFF);
        guest_copy(0, 0, 64, 64, GuestBase, false, true, 7);
        setup_2d(false);
        GXSetTevColor(GX_TEVREG0, Blue);
        quad(16, 16, 32, 32);
      },
      AURORA_READBACK_PEEK_ARGB, {{20, 20}, {40, 40}});
  EXPECT_EQ(v[0] & 0xFFFFFF, 0x0000FFu);
  EXPECT_EQ(v[1] & 0xFFFFFF, 0xFF0000u); // the colour is kept
  EXPECT_EQ(v[1] >> 24, 0u);             // the alpha cleared
}
