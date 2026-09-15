// Loads bmp-2048x1365.bmp (whose own rows happen to need no BMP padding --
// 2048 * 3 channels = 6144, already a multiple of 4), copies it into a
// buffer padded to a stride wider than width * channels -- the shape of a
// real hardware surface with alignment padding -- and runs three copies
// off that padded source: one using the real stride (correct), one
// substituting width for stride (silently sheared, no crash), and one that
// under-sizes the destination and writes at the source's stride anyway
// (out-of-bounds write).
//
// Build and run:
//   g++ -O2 -std=c++23 stride_demo.cpp -o stride_demo `pkg-config --cflags --libs opencv4`
//   ./stride_demo                 # correct + sheared demos only
//   ./stride_demo --trigger-overrun   # also runs the out-of-bounds write

#include "stride_view.hpp"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{

// A fixed per-row pad rather than an alignment round-up: 2048 * 3 channels
// is already a multiple of every alignment up to 2048 bytes, so rounding
// up to a typical pitch alignment (128, 256...) would silently produce
// zero padding for this specific asset and demonstrate nothing.
constexpr std::size_t kRowPaddingBytes = 64;

std::size_t padded_stride(int width, int channels)
{
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(channels) + kRowPaddingBytes;
}

// Copies `img` into a newly allocated buffer with rows padded to `stride`
// bytes -- the padding bytes are left non-zero (filled with a marker) so a
// sheared read visibly picks up wrong data instead of silently reading zeros.
std::vector<std::uint8_t> make_padded_source(const cv::Mat &img, std::size_t stride)
{
    const int height = img.rows;
    const int width = img.cols;
    const int channels = img.channels();
    const std::size_t row_bytes = static_cast<std::size_t>(width) * channels;

    std::vector<std::uint8_t> buf(stride * static_cast<std::size_t>(height), 0xAA);
    for (int y = 0; y < height; ++y)
    {
        std::memcpy(buf.data() + static_cast<std::size_t>(y) * stride, img.ptr<std::uint8_t>(y), row_bytes);
    }
    return buf;
}

} // namespace

int main(int argc, char **argv)
{
    bool trigger_overrun = (argc > 1 && std::string(argv[1]) == "--trigger-overrun");

    cv::Mat img = cv::imread("bmp-2048x1365.bmp", cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::fprintf(stderr, "could not load bmp-2048x1365.bmp (run from ai-cpp-l2/)\n");
        return 1;
    }

    const int width = img.cols;
    const int height = img.rows;
    const int channels = img.channels();
    const std::size_t row_bytes = static_cast<std::size_t>(width) * channels;
    const std::size_t stride = padded_stride(width, channels);
    std::printf("image %dx%d, %d channels: row_bytes=%zu, padded stride=%zu (padding=%zu bytes/row)\n",
                width, height, channels, row_bytes, stride, stride - row_bytes);

    auto padded = make_padded_source(img, stride);
    stride_lesson::View src{padded.data(), width, height, channels, stride};

    // --- Correct: byte-identical to the source image ---
    std::vector<std::uint8_t> correct_dst(row_bytes * static_cast<std::size_t>(height));
    stride_lesson::copy_using_stride(src, correct_dst.data());
    bool matches = std::memcmp(correct_dst.data(), img.data, correct_dst.size()) == 0;
    std::printf("correct (uses real stride):  %s\n", matches ? "matches source exactly" : "MISMATCH (bug in copy_using_stride)");

    // --- Sheared: width substituted for stride, no crash, wrong content ---
    std::vector<std::uint8_t> sheared_dst(row_bytes * static_cast<std::size_t>(height));
    stride_lesson::copy_using_width_as_stride(src, sheared_dst.data());
    std::size_t first_row_diff = 0;
    for (; first_row_diff < static_cast<std::size_t>(height); ++first_row_diff)
    {
        if (std::memcmp(sheared_dst.data() + first_row_diff * row_bytes,
                         img.data + first_row_diff * row_bytes, row_bytes) != 0)
        {
            break;
        }
    }
    std::printf("sheared (width used as stride): first differing row = %zu (of %d) -- silent, no crash\n",
                first_row_diff, height);

    // --- Overrun: dst sized for row_bytes, written at src's stride ---
    if (trigger_overrun)
    {
        std::printf("triggering overrun: dst allocated for %zu bytes, write reaches offset %zu\n",
                    row_bytes * static_cast<std::size_t>(height),
                    static_cast<std::size_t>(height - 1) * stride + row_bytes);
        auto dst = std::make_unique<std::uint8_t[]>(row_bytes * static_cast<std::size_t>(height));
        stride_lesson::copy_overrunning_dst(src, dst.get(), row_bytes * static_cast<std::size_t>(height));
        std::printf("overrun copy returned without a sanitizer catching it -- rebuild with ASan\n");
    }

    return 0;
}
