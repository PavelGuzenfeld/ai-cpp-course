#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <cstdint>

namespace nb = nanobind;

using ImageIn = nb::ndarray<nb::numpy, const float, nb::ndim<2>>;
using MaskIn = nb::ndarray<nb::numpy, const float, nb::ndim<2>>;

// Linear stage: a naive (2*radius+1)^2 box blur. Hand-rolled on purpose —
// its golden-oracle test (test_validation.py) checks it against an
// independently-ordered NumPy summation, not against its own formula.
nb::ndarray<nb::numpy, float> box_blur(ImageIn image, int radius)
{
    auto height = static_cast<int>(image.shape(0));
    auto width = static_cast<int>(image.shape(1));
    float *out = new float[static_cast<std::size_t>(height) * width];

    {
        // Scoped release: the capsule/ndarray built below are Python
        // objects and must not be constructed while the GIL is released.
        nb::gil_scoped_release release;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float sum = 0.0F;
                int count = 0;
                for (int dy = -radius; dy <= radius; ++dy)
                {
                    int sy = std::clamp(y + dy, 0, height - 1);
                    for (int dx = -radius; dx <= radius; ++dx)
                    {
                        int sx = std::clamp(x + dx, 0, width - 1);
                        sum += image(sy, sx);
                        ++count;
                    }
                }
                out[y * width + x] = sum / static_cast<float>(count);
            }
        }
    }

    std::size_t shape[2] = {static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    return nb::ndarray<nb::numpy, float>(out, 2, shape, owner);
}

// Thresholded stage: a hard nonlinearity. Its test (test_validation.py) uses
// mask-disagreement rate rather than max-abs-diff — see that file for why.
nb::ndarray<nb::numpy, bool> threshold_mask(MaskIn image, float threshold)
{
    auto height = static_cast<int>(image.shape(0));
    auto width = static_cast<int>(image.shape(1));
    bool *out = new bool[static_cast<std::size_t>(height) * width];

    {
        nb::gil_scoped_release release;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                out[y * width + x] = image(y, x) > threshold;
            }
        }
    }

    std::size_t shape[2] = {static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<bool *>(p); });
    return nb::ndarray<nb::numpy, bool>(out, 2, shape, owner);
}

NB_MODULE(filter_native, m)
{
    m.doc() = "Box blur (linear) and threshold (nonlinear) stages for the L18 golden-oracle lesson";

    m.def("box_blur", &box_blur, nb::arg("image"), nb::arg("radius"),
          "Naive box blur with clamp-to-edge boundary handling");

    m.def("threshold_mask", &threshold_mask, nb::arg("image"), nb::arg("threshold"),
          "Per-pixel threshold, image > threshold");
}
