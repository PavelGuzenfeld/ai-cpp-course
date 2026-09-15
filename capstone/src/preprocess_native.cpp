/**
 * preprocess_native: a single fused pass over the pixel data -- cast,
 * normalize and transpose (HWC uint8 -> CHW float32 in [0,1]) in one loop
 * instead of the baseline's three separate NumPy passes.
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>

namespace nb = nanobind;

class FusedPreprocessor
{
public:
    nb::ndarray<nb::numpy, float> process(
        nb::ndarray<const std::uint8_t, nb::ndim<3>> image)
    {
        auto height = static_cast<int>(image.shape(0));
        auto width = static_cast<int>(image.shape(1));
        auto channels = static_cast<int>(image.shape(2));

        auto *out = new float[static_cast<std::size_t>(channels) * height * width];
        const std::uint8_t *src = image.data();
        constexpr float kInv255 = 1.0F / 255.0F; // avoid a per-element float divide

        // Channel outermost so the x-loop (the vectorizable one, since
        // `channels` is a runtime value the compiler can't unroll) writes
        // one contiguous output plane per pass. A channel-innermost
        // "deinterleave" order was tried and measured slower: it moves the
        // vectorizable loop to the outside and leaves the compiler unable
        // to auto-vectorize the tiny runtime-trip-count channel loop.
        for (int c = 0; c < channels; ++c)
        {
            float *out_plane = out + static_cast<std::size_t>(c) * height * width;
            for (int y = 0; y < height; ++y)
            {
                const std::uint8_t *src_row = src + (static_cast<std::size_t>(y) * width) * channels + c;
                for (int x = 0; x < width; ++x)
                {
                    out_plane[y * width + x] = static_cast<float>(src_row[x * channels]) * kInv255;
                }
            }
        }

        std::size_t shape[3] = {
            static_cast<std::size_t>(channels),
            static_cast<std::size_t>(height),
            static_cast<std::size_t>(width)};
        nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<float *>(p); });
        return nb::ndarray<nb::numpy, float>(out, 3, shape, owner);
    }
};

NB_MODULE(preprocess_native, m)
{
    m.doc() = "Fused HWC uint8 -> CHW float32 [0,1] preprocessing (capstone component 2)";

    nb::class_<FusedPreprocessor>(m, "FusedPreprocessor")
        .def(nb::init<>())
        .def("process", &FusedPreprocessor::process, nb::arg("image"));
}
