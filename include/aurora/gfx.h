#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#endif

#if !defined(NDEBUG) && !defined(AURORA_GFX_DEBUG_GROUPS)
#define AURORA_GFX_DEBUG_GROUPS
#endif

void push_debug_group(const char* label);
void pop_debug_group();

typedef struct {
  uint32_t queuedPipelines;
  uint32_t createdPipelines;
  uint32_t drawCallCount;
  uint32_t mergedDrawCallCount;
  uint32_t lastVertSize;
  uint32_t lastUniformSize;
  uint32_t lastIndexSize;
  uint32_t lastStorageSize;
  uint32_t lastTextureUploadSize;
  /* Cumulative since initialisation (any thread): pipelines compiled and the time spent compiling them, draws that
     waited for their pipeline (blocking draws) and the time they waited, draws skipped because their pipeline was
     not ready (async pipelines). */
  uint32_t pipelineWaits;
  uint32_t skippedDraws;
  uint64_t pipelineCompileNs;
  uint64_t pipelineWaitNs;
} AuroraStats;

const AuroraStats* aurora_get_stats();
float aurora_get_fps();

void aurora_enable_vsync(bool enabled);

/**
 * The adapters offered for the selected backend, one per line ("index: name (type, backend, driver)"), as
 * AuroraConfig::adapterName counts them. Copies at most size - 1 bytes plus a terminator; returns the full length.
 */
size_t aurora_get_adapter_list(char* buffer, size_t size);

/**
 * Display aspect ratio (width / height) the main framebuffer is presented at. The framebuffer is
 * letterboxed or pillarboxed into the window at this aspect instead of its own pixel aspect, as a TV
 * shows a 4:3 or anamorphic 16:9 picture from the same framebuffer. Zero (the default) presents
 * with square pixels.
 */
void aurora_set_present_aspect(float aspect);

/**
 * Reads back the color contents of the main framebuffer (EFB) as of the last aurora_end_frame(), as RGBA8 rows from
 * top to bottom (width * 4 bytes per row). Waits for all submitted GPU work, so it is meant for tests and
 * screenshots, not for use every frame. width and height (optional) receive the framebuffer size even on failure.
 * Returns false if there is no framebuffer, its format is not 8-bit RGBA/BGRA, or dstSize is too small.
 */
bool aurora_read_framebuffer(void* dst, size_t dstSize, uint32_t* width, uint32_t* height);

#ifdef __cplusplus
}
#endif

#endif
