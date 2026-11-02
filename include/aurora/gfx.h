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
 * With vsync, prefer the mailbox present mode where the surface offers it: frames are still shown whole at a
 * refresh, but presenting never waits for one (the application paces itself; the newest frame is shown). Off by
 * default (FIFO relaxed / FIFO).
 */
void aurora_set_vsync_mailbox(bool mailbox);

/** The present mode of the main window ("fifo", "fifo-relaxed", "mailbox", "immediate"; "none" headless). */
const char* aurora_get_present_mode();

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

/** Where the last presented frame went: the viewport inside the surface (pixels) and the sizes it was presented at. */
typedef struct AuroraPresentViewport {
  float left;
  float top;
  float width;
  float height;
  uint32_t surfaceWidth;  /* the surface (window) in pixels */
  uint32_t surfaceHeight;
  uint32_t windowWidth;   /* the window in logical units (points) at that time */
  uint32_t windowHeight;
  bool valid;
} AuroraPresentViewport;

/**
 * The viewport and surface size of the last presentation (the letterbox / pillarbox the picture was drawn into), for
 * mapping window coordinates (mouse) onto the picture. Returns false before the first presentation.
 */
bool aurora_get_present_viewport(AuroraPresentViewport* out);

/**
 * Limits the frames in flight: aurora_begin_frame() waits until fewer than `frames` ended frames have rendering
 * outstanding on the GPU. Their presentation is not waited for: acquiring the surface's next texture waits for the
 * compositor, which would put the caller in step with it. 0 (the default) leaves only the internal frame slots and
 * staging buffers as a bound. 1 waits for the previous frame's rendering before starting the next.
 */
void aurora_set_max_frames_in_flight(uint32_t frames);

typedef struct AuroraFrameStats {
  uint64_t framesEnded;      /* aurora_end_frame calls */
  uint64_t framesDone;       /* frames whose rendering has completed */
  uint64_t lastGpuLatencyNs; /* the latest presented frame: from aurora_end_frame to its GPU work (presentation
                                included) done, as seen by the render thread */
  uint64_t gpuLatencySumNs;  /* sum over the completed frames */
  uint64_t gpuLatencyMaxNs;
  uint64_t inFlightWaitNs;   /* time aurora_begin_frame waited for the frames-in-flight limit */
  uint32_t maxInFlightSeen;  /* the most frames outstanding at an aurora_end_frame */
  uint32_t maxFramesInFlight;
  /* The latest frame on the render thread, from aurora_end_frame: until its recording started (the queue), until
     its rendering (presentation excluded) was done on the GPU */
  uint64_t lastQueueLatencyNs;
  uint64_t lastRenderLatencyNs;
  /* Cumulative render-thread time acquiring the surface's next texture and in Surface::Present */
  uint64_t acquireNs;
  uint64_t presentNs;
} AuroraFrameStats;
void aurora_get_frame_stats(AuroraFrameStats* out);

/** Tests: adds `repeats` copies of a `mib` MiB buffer to every frame's GPU work (a slow GPU). 0 turns it off. */
void aurora_test_set_gpu_load(uint32_t mib, uint32_t repeats);

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
