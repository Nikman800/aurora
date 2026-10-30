#pragma once

#include "../internal.hpp"

#include <cstdint>

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;

// Address resolver for 32-bit addresses in the command stream (AuroraSetAddressResolver).
using AddressResolver = const void* (*)(uint32_t address, uint32_t* size, void* userData);
void set_address_resolver(AddressResolver resolver, void* userData) noexcept;
bool has_address_resolver() noexcept;
// Host pointer for `address`, or nullptr if it is unmapped or fewer than `minSize` bytes
// are readable; `available` (optional) receives the readable size.
const uint8_t* resolve_address(uint32_t address, uint32_t minSize, uint32_t* available = nullptr) noexcept;

} // namespace aurora::gx::fifo
