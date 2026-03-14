/**
 * CPU reference implementation of fused preprocessing.
 *
 * Same interface as gpu_preprocess.cu but uses std::transform on CPU.
 * Used for:
 *   1. Correctness validation (compare GPU output against this)
 *   2. The "before" benchmark (this is what tracker_engine effectively does)
 *   3. Fallback when CUDA is not available
 */

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

/**
 * CPU fused preprocess: uint8 HWC → float32 CHW + normalize.
 *
 * Uses std::transform for the element-wise conversion, mimicking what
 * tracker_engine does with numpy but in C++ for a fair comparison.
 */
nb::ndarray<nb::numpy, float> fused_preprocess(
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<3>> input,
    std::vector<float> mean,
    std::vector<float> std_dev)
{
    int height = static_cast<int>(input.shape(0));
    int width = static_cast<int>(input.shape(1));
    int channels = static_cast<int>(input.shape(2));
    int total = height * width * channels;

    if (mean.size() != static_cast<size_t>(channels) ||
        std_dev.size() != static_cast<size_t>(channels)) {
        throw std::invalid_argument("mean and std must have length == channels");
    }

    float* output = new float[total];
    const uint8_t* src = input.data();

    // Fused HWC→CHW transpose + normalize in a single pass
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                float pixel = static_cast<float>(src[h * width * channels + w * channels + c]);
                pixel = (pixel / 255.0f - mean[c]) / std_dev[c];
                output[c * height * width + h * width + w] = pixel;
            }
        }
    }

    size_t shape[3] = {
        static_cast<size_t>(channels),
        static_cast<size_t>(height),
        static_cast<size_t>(width)
    };
    nb::capsule owner(output, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float>(output, 3, shape, owner);
}

/**
 * CPU numpy-style preprocess (mimics tracker_engine's TRT_Preprocessor.process).
 *
 * This is the "naive" approach for benchmarking — same three-step process:
 *   1. Cast to float32
 *   2. Normalize with mean/std
 *   3. Transpose HWC → CHW
 */
nb::ndarray<nb::numpy, float> numpy_style_preprocess(
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<3>> input,
    std::vector<float> mean,
    std::vector<float> std_dev)
{
    int height = static_cast<int>(input.shape(0));
    int width = static_cast<int>(input.shape(1));
    int channels = static_cast<int>(input.shape(2));
    int total = height * width * channels;

    if (mean.size() != static_cast<size_t>(channels) ||
        std_dev.size() != static_cast<size_t>(channels)) {
        throw std::invalid_argument("mean and std must have length == channels");
    }

    const uint8_t* src = input.data();

    // Step 1: Cast to float32 (like image.astype(np.float32))
    std::vector<float> float_buf(total);
    std::transform(src, src + total, float_buf.begin(),
        [](uint8_t v) { return static_cast<float>(v); });

    // Step 2: Normalize (like (image / 255.0 - mean) / std)
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                int idx = h * width * channels + w * channels + c;
                float_buf[idx] = (float_buf[idx] / 255.0f - mean[c]) / std_dev[c];
            }
        }
    }

    // Step 3: Transpose HWC → CHW (like image.transpose(2, 0, 1))
    float* output = new float[total];
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                output[c * height * width + h * width + w] =
                    float_buf[h * width * channels + w * channels + c];
            }
        }
    }

    size_t shape[3] = {
        static_cast<size_t>(channels),
        static_cast<size_t>(height),
        static_cast<size_t>(width)
    };
    nb::capsule owner(output, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float>(output, 3, shape, owner);
}

bool cuda_available() { return false; }

// ─── Nanobind module ─────────────────────────────────────────────────────────

#ifndef MODULE_NAME
#define MODULE_NAME gpu_preprocess
#endif

NB_MODULE(MODULE_NAME, m) {
    m.doc() = "CPU reference implementation of fused preprocessing";

    m.def("fused_preprocess", &fused_preprocess,
          nb::arg("input"), nb::arg("mean"), nb::arg("std"),
          "CPU fused preprocess: uint8 HWC → float32 CHW normalized.");

    m.def("numpy_style_preprocess", &numpy_style_preprocess,
          nb::arg("input"), nb::arg("mean"), nb::arg("std"),
          "Three-step CPU preprocess mimicking tracker_engine's approach.");

    m.def("cuda_available", &cuda_available,
          "Always returns False for CPU reference build.");
}
