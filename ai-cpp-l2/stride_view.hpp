/**
 * A row-major image view that carries its real stride (bytes per row)
 * separately from width * channels. On real hardware buffers the two are
 * rarely equal -- rows are padded to an alignment the format descriptor
 * does not mention, and code that reinvents row-start as `y * width *
 * channels` reads or writes the wrong bytes without any indication it did.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace stride_lesson
{

struct View
{
    const std::uint8_t *data;
    int width;
    int height;
    int channels;
    std::size_t stride; // bytes between the start of row y and row y+1

    [[nodiscard]] const std::uint8_t *row(int y) const
    {
        return data + static_cast<std::size_t>(y) * stride;
    }

    [[nodiscard]] std::size_t row_bytes() const
    {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    }
};

/**
 * Copies `src` into a tightly-packed `dst` (no padding), one row at a time,
 * reading each row at its real stride. This is the correct version: the
 * output is byte-identical to what `src` represents regardless of how much
 * padding its rows carry.
 */
inline void copy_using_stride(const View &src, std::uint8_t *dst)
{
    const std::size_t row_bytes = src.row_bytes();
    for (int y = 0; y < src.height; ++y)
    {
        std::memcpy(dst + static_cast<std::size_t>(y) * row_bytes, src.row(y), row_bytes);
    }
}

/**
 * The bug: assumes row `y` starts at `y * width * channels`, ignoring
 * `src.stride`. When stride > row_bytes (the padded case), every row after
 * the first is read from the wrong offset -- each row picks up bytes that
 * belong to the tail of the previous row plus its padding. The result is a
 * sheared image: same size, wrong content, no crash, nothing in the return
 * value says anything went wrong.
 */
inline void copy_using_width_as_stride(const View &src, std::uint8_t *dst)
{
    const std::size_t row_bytes = src.row_bytes();
    for (int y = 0; y < src.height; ++y)
    {
        const std::uint8_t *wrong_row = src.data + static_cast<std::size_t>(y) * row_bytes;
        std::memcpy(dst + static_cast<std::size_t>(y) * row_bytes, wrong_row, row_bytes);
    }
}

/**
 * The other half of the bug: `dst` is sized as if it had no padding
 * (`width * height * channels`, exactly `dst_capacity` bytes below), but
 * the copy loop writes each row at `src`'s real stride -- reusing the
 * source's row spacing for a destination that was never allocated to have
 * it. Once `stride > row_bytes`, the write offset for row y exceeds what
 * `dst_capacity` bytes actually holds well before the last row, and the
 * final row's write runs past the end of the allocation entirely. This is
 * the shape of the gst-nvmm-cpp bug this lesson is drawn from: the
 * corrupted heap byte is nowhere near the write that corrupted it by the
 * time anything crashes, unless a sanitizer is watching the write itself.
 */
inline void copy_overrunning_dst(const View &src, std::uint8_t *dst, std::size_t dst_capacity)
{
    const std::size_t row_bytes = src.row_bytes();
    (void)dst_capacity; // the bug is that nothing here checks against it
    for (int y = 0; y < src.height; ++y)
    {
        std::memcpy(dst + static_cast<std::size_t>(y) * src.stride, src.row(y), row_bytes);
    }
}

} // namespace stride_lesson
