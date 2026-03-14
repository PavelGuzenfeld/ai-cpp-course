#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Image template with compile-time dimensions
// ---------------------------------------------------------------------------
template <int W, int H, int C>
struct Image {
    static constexpr int kWidth = W;
    static constexpr int kHeight = H;
    static constexpr int kChannels = C;
    static constexpr size_t kSize = static_cast<size_t>(W) * H * C;

    std::array<uint8_t, kSize> pixels{};

    [[nodiscard]] const uint8_t* data() const noexcept { return pixels.data(); }
    [[nodiscard]] uint8_t* data() noexcept { return pixels.data(); }
    [[nodiscard]] constexpr int width() const noexcept { return W; }
    [[nodiscard]] constexpr int height() const noexcept { return H; }
    [[nodiscard]] constexpr int channels() const noexcept { return C; }
};

// ---------------------------------------------------------------------------
// Compile-time BGR-to-Grayscale LUT
//
// Standard BT.601 weights: Gray = 0.114*B + 0.587*G + 0.299*R
// We store the weighted contribution of each channel value (0..255)
// in three sub-tables of 256 entries each.
// ---------------------------------------------------------------------------
constexpr auto make_grayscale_lut() {
    std::array<uint8_t, 256 * 3> lut{};
    for (int i = 0; i < 256; ++i) {
        // B channel contribution (index 0..255)
        lut[i] = static_cast<uint8_t>(i * 0.114);
        // G channel contribution (index 256..511)
        lut[256 + i] = static_cast<uint8_t>(i * 0.587);
        // R channel contribution (index 512..767)
        lut[512 + i] = static_cast<uint8_t>(i * 0.299);
    }
    return lut;
}

// The LUT is embedded in the binary's .rodata section at compile time.
constexpr auto GRAY_LUT = make_grayscale_lut();

// ---------------------------------------------------------------------------
// Compile-time Gamma Correction LUT
//
// std::pow is not constexpr, so we use an iterative Newton's method
// approximation for x^gamma.
// ---------------------------------------------------------------------------
constexpr double constexpr_pow(double base, double exp) {
    if (base <= 0.0) return 0.0;
    if (exp == 0.0) return 1.0;
    if (exp == 1.0) return base;

    // Use exp(exp * ln(base)) approximation via iterative method.
    // ln(base) via series: ln(x) = 2 * sum_{k=0}^{N} (1/(2k+1)) * ((x-1)/(x+1))^(2k+1)
    double t = (base - 1.0) / (base + 1.0);
    double t2 = t * t;
    double ln_base = 0.0;
    double term = t;
    for (int k = 0; k < 40; ++k) {
        ln_base += term / (2.0 * k + 1.0);
        term *= t2;
    }
    ln_base *= 2.0;

    // exp(y) via Taylor series
    double y = exp * ln_base;
    double result = 1.0;
    double factorial_term = 1.0;
    for (int k = 1; k <= 30; ++k) {
        factorial_term *= y / k;
        result += factorial_term;
    }
    return result;
}

constexpr auto make_gamma_lut(double gamma) {
    std::array<uint8_t, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        double normalized = i / 255.0;
        double corrected = constexpr_pow(normalized, gamma);
        int val = static_cast<int>(corrected * 255.0 + 0.5);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        lut[i] = static_cast<uint8_t>(val);
    }
    return lut;
}

// Common gamma values, precomputed at compile time.
constexpr auto GAMMA_LUT_0_45 = make_gamma_lut(0.45);   // sRGB encoding gamma
constexpr auto GAMMA_LUT_2_2 = make_gamma_lut(2.2);     // sRGB decoding gamma
constexpr auto GAMMA_LUT_1_0 = make_gamma_lut(1.0);     // Identity (for testing)

// ---------------------------------------------------------------------------
// Runtime grayscale conversion (for benchmarking comparison)
// ---------------------------------------------------------------------------
void grayscale_runtime(const uint8_t* bgr, uint8_t* gray, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; ++i) {
        uint8_t b = bgr[i * 3 + 0];
        uint8_t g = bgr[i * 3 + 1];
        uint8_t r = bgr[i * 3 + 2];
        gray[i] = static_cast<uint8_t>(b * 0.114 + g * 0.587 + r * 0.299);
    }
}

// ---------------------------------------------------------------------------
// LUT-based grayscale conversion
// ---------------------------------------------------------------------------
void grayscale_lut(const uint8_t* bgr, uint8_t* gray, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; ++i) {
        uint8_t b = bgr[i * 3 + 0];
        uint8_t g = bgr[i * 3 + 1];
        uint8_t r = bgr[i * 3 + 2];
        gray[i] = GRAY_LUT[b] + GRAY_LUT[256 + g] + GRAY_LUT[512 + r];
    }
}

// ---------------------------------------------------------------------------
// Runtime gamma correction (for benchmarking comparison)
// ---------------------------------------------------------------------------
void gamma_runtime(const uint8_t* src, uint8_t* dst, size_t num_pixels, double gamma) {
    for (size_t i = 0; i < num_pixels; ++i) {
        double normalized = src[i] / 255.0;
        double corrected = std::pow(normalized, gamma);
        int val = static_cast<int>(corrected * 255.0 + 0.5);
        dst[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
    }
}

// ---------------------------------------------------------------------------
// LUT-based gamma correction
// ---------------------------------------------------------------------------
void gamma_lut(const uint8_t* src, uint8_t* dst, size_t num_pixels,
               const std::array<uint8_t, 256>& lut) {
    for (size_t i = 0; i < num_pixels; ++i) {
        dst[i] = lut[src[i]];
    }
}

// ---------------------------------------------------------------------------
// Nanobind bindings
// ---------------------------------------------------------------------------
NB_MODULE(compile_time_lut, m) {
    m.doc() = "Compile-time LUT generation for grayscale and gamma correction";

    // --- Apply grayscale LUT to a BGR numpy array, return grayscale array ---
    m.def("apply_grayscale_lut", [](nb::ndarray<uint8_t, nb::ndim<3>> bgr_arr)
          -> nb::ndarray<nb::numpy, uint8_t> {
        int h = static_cast<int>(bgr_arr.shape(0));
        int w = static_cast<int>(bgr_arr.shape(1));
        int c = static_cast<int>(bgr_arr.shape(2));
        if (c != 3) {
            throw std::invalid_argument("Expected 3-channel BGR image");
        }
        size_t num_pixels = static_cast<size_t>(h) * w;
        auto* out = new uint8_t[num_pixels];
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });

        const uint8_t* src = bgr_arr.data();
        grayscale_lut(src, out, num_pixels);

        size_t shape[2] = {static_cast<size_t>(h), static_cast<size_t>(w)};
        return nb::ndarray<nb::numpy, uint8_t>(out, 2, shape, owner);
    }, nb::arg("bgr"), "Convert BGR image to grayscale using compile-time LUT");

    // --- Apply grayscale runtime to a BGR numpy array ---
    m.def("apply_grayscale_runtime", [](nb::ndarray<uint8_t, nb::ndim<3>> bgr_arr)
          -> nb::ndarray<nb::numpy, uint8_t> {
        int h = static_cast<int>(bgr_arr.shape(0));
        int w = static_cast<int>(bgr_arr.shape(1));
        int c = static_cast<int>(bgr_arr.shape(2));
        if (c != 3) {
            throw std::invalid_argument("Expected 3-channel BGR image");
        }
        size_t num_pixels = static_cast<size_t>(h) * w;
        auto* out = new uint8_t[num_pixels];
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });

        const uint8_t* src = bgr_arr.data();
        grayscale_runtime(src, out, num_pixels);

        size_t shape[2] = {static_cast<size_t>(h), static_cast<size_t>(w)};
        return nb::ndarray<nb::numpy, uint8_t>(out, 2, shape, owner);
    }, nb::arg("bgr"), "Convert BGR image to grayscale using runtime computation");

    // --- Apply gamma LUT ---
    m.def("apply_gamma_lut", [](nb::ndarray<uint8_t, nb::ndim<2>> gray_arr, double gamma)
          -> nb::ndarray<nb::numpy, uint8_t> {
        int h = static_cast<int>(gray_arr.shape(0));
        int w = static_cast<int>(gray_arr.shape(1));
        size_t num_pixels = static_cast<size_t>(h) * w;
        auto* out = new uint8_t[num_pixels];
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });

        const uint8_t* src = gray_arr.data();

        // Select precomputed LUT if available, otherwise fall back to runtime-built LUT
        if (gamma == 0.45) {
            gamma_lut(src, out, num_pixels, GAMMA_LUT_0_45);
        } else if (gamma == 2.2) {
            gamma_lut(src, out, num_pixels, GAMMA_LUT_2_2);
        } else if (gamma == 1.0) {
            gamma_lut(src, out, num_pixels, GAMMA_LUT_1_0);
        } else {
            // Build LUT at runtime for non-standard gamma values
            auto custom_lut = make_gamma_lut(gamma);
            gamma_lut(src, out, num_pixels, custom_lut);
        }

        size_t shape[2] = {static_cast<size_t>(h), static_cast<size_t>(w)};
        return nb::ndarray<nb::numpy, uint8_t>(out, 2, shape, owner);
    }, nb::arg("gray"), nb::arg("gamma"),
    "Apply gamma correction using compile-time LUT (precomputed for gamma=0.45, 2.2, 1.0)");

    // --- Apply gamma runtime ---
    m.def("apply_gamma_runtime", [](nb::ndarray<uint8_t, nb::ndim<2>> gray_arr, double gamma)
          -> nb::ndarray<nb::numpy, uint8_t> {
        int h = static_cast<int>(gray_arr.shape(0));
        int w = static_cast<int>(gray_arr.shape(1));
        size_t num_pixels = static_cast<size_t>(h) * w;
        auto* out = new uint8_t[num_pixels];
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });

        const uint8_t* src = gray_arr.data();
        gamma_runtime(src, out, num_pixels, gamma);

        size_t shape[2] = {static_cast<size_t>(h), static_cast<size_t>(w)};
        return nb::ndarray<nb::numpy, uint8_t>(out, 2, shape, owner);
    }, nb::arg("gray"), nb::arg("gamma"),
    "Apply gamma correction using runtime pow() computation");

    // --- Expose LUT data for inspection ---
    m.def("get_grayscale_lut_value", [](int channel, int intensity) -> uint8_t {
        if (channel < 0 || channel > 2 || intensity < 0 || intensity > 255) {
            throw std::out_of_range("channel must be 0-2, intensity must be 0-255");
        }
        return GRAY_LUT[channel * 256 + intensity];
    }, nb::arg("channel"), nb::arg("intensity"),
    "Get a single value from the compile-time grayscale LUT (channel: 0=B, 1=G, 2=R)");

    m.def("get_gamma_lut_value", [](int intensity, double gamma) -> uint8_t {
        if (intensity < 0 || intensity > 255) {
            throw std::out_of_range("intensity must be 0-255");
        }
        if (gamma == 0.45) return GAMMA_LUT_0_45[intensity];
        if (gamma == 2.2) return GAMMA_LUT_2_2[intensity];
        if (gamma == 1.0) return GAMMA_LUT_1_0[intensity];
        throw std::invalid_argument("Only precomputed gamma values (0.45, 2.2, 1.0) available");
    }, nb::arg("intensity"), nb::arg("gamma"),
    "Get a single value from a compile-time gamma LUT");
}
