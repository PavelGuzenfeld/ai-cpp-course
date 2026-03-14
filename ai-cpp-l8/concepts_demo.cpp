#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <array>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Concept: FlatType
// A type that can be safely memcpy'd, mapped into shared memory, or sent to a
// GPU. This is the gatekeeper for safe-shm (Lesson 3).
// ---------------------------------------------------------------------------
template <typename T>
concept FlatType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

// ---------------------------------------------------------------------------
// Concept: Numeric
// Any arithmetic type (int, float, double, etc.)
// ---------------------------------------------------------------------------
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

// ---------------------------------------------------------------------------
// Concept: ImageLike
// Structural constraint: anything with data(), width(), height(), channels()
// ---------------------------------------------------------------------------
template <typename T>
concept ImageLike = requires(const T img) {
    { img.data() } -> std::convertible_to<const void*>;
    { img.width() } -> std::convertible_to<int>;
    { img.height() } -> std::convertible_to<int>;
    { img.channels() } -> std::convertible_to<int>;
};

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
    [[nodiscard]] constexpr size_t size() const noexcept { return kSize; }

    uint8_t& at(int x, int y, int c) {
        return pixels[static_cast<size_t>(y) * W * C + static_cast<size_t>(x) * C + c];
    }

    [[nodiscard]] const uint8_t& at(int x, int y, int c) const {
        return pixels[static_cast<size_t>(y) * W * C + static_cast<size_t>(x) * C + c];
    }
};

// ---------------------------------------------------------------------------
// Flat structs that satisfy FlatType
// ---------------------------------------------------------------------------
struct BBox {
    float x, y, w, h;
};

struct Point2D {
    double x, y;
};

struct TrackingResult {
    int id;
    float confidence;
    BBox bbox;
};

// ---------------------------------------------------------------------------
// Non-flat types (for demonstration of concept rejection)
// ---------------------------------------------------------------------------
// std::string is NOT FlatType — it contains a heap pointer
// std::vector<T> is NOT FlatType — it contains a heap pointer

// ---------------------------------------------------------------------------
// Constrained function: serialize only FlatType
// ---------------------------------------------------------------------------
template <FlatType T>
std::vector<uint8_t> serialize(const T& obj) {
    std::vector<uint8_t> buf(sizeof(T));
    std::memcpy(buf.data(), &obj, sizeof(T));
    return buf;
}

template <FlatType T>
T deserialize(const std::vector<uint8_t>& buf) {
    if (buf.size() < sizeof(T)) {
        throw std::runtime_error("buffer too small for deserialization");
    }
    T obj;
    std::memcpy(&obj, buf.data(), sizeof(T));
    return obj;
}

// ---------------------------------------------------------------------------
// Constrained function: process only ImageLike types
// ---------------------------------------------------------------------------
template <ImageLike Img>
int total_pixels(const Img& img) {
    return img.width() * img.height();
}

template <ImageLike Img>
size_t total_bytes(const Img& img) {
    return static_cast<size_t>(img.width()) * img.height() * img.channels();
}

// ---------------------------------------------------------------------------
// Constrained function: numeric operations
// ---------------------------------------------------------------------------
template <Numeric T>
T clamp_value(T val, T lo, T hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

// ---------------------------------------------------------------------------
// Compile-time concept checks (exposed to Python as booleans)
// ---------------------------------------------------------------------------
constexpr bool int_is_flat = FlatType<int>;
constexpr bool float_is_flat = FlatType<float>;
constexpr bool double_is_flat = FlatType<double>;
constexpr bool bbox_is_flat = FlatType<BBox>;
constexpr bool point2d_is_flat = FlatType<Point2D>;
constexpr bool tracking_result_is_flat = FlatType<TrackingResult>;
constexpr bool string_is_flat = FlatType<std::string>;
constexpr bool vector_int_is_flat = FlatType<std::vector<int>>;

// Image types satisfy ImageLike
using SmallRGB = Image<64, 48, 3>;
using SmallGray = Image<64, 48, 1>;
using HD_RGB = Image<1920, 1080, 3>;

constexpr bool small_rgb_is_imagelike = ImageLike<SmallRGB>;
constexpr bool small_gray_is_imagelike = ImageLike<SmallGray>;

// ---------------------------------------------------------------------------
// Nanobind bindings
// ---------------------------------------------------------------------------
NB_MODULE(concepts_demo, m) {
    m.doc() = "Demonstrates C++20 concepts: FlatType, Numeric, ImageLike";

    // --- FlatType checks ---
    m.attr("int_is_flat") = int_is_flat;
    m.attr("float_is_flat") = float_is_flat;
    m.attr("double_is_flat") = double_is_flat;
    m.attr("bbox_is_flat") = bbox_is_flat;
    m.attr("point2d_is_flat") = point2d_is_flat;
    m.attr("tracking_result_is_flat") = tracking_result_is_flat;
    m.attr("string_is_flat") = string_is_flat;
    m.attr("vector_int_is_flat") = vector_int_is_flat;
    m.attr("image_is_imagelike") = small_rgb_is_imagelike;

    // --- BBox binding ---
    nb::class_<BBox>(m, "BBox")
        .def(nb::init<>())
        .def_rw("x", &BBox::x)
        .def_rw("y", &BBox::y)
        .def_rw("w", &BBox::w)
        .def_rw("h", &BBox::h)
        .def("__repr__", [](const BBox& b) {
            return "BBox(x=" + std::to_string(b.x) + ", y=" + std::to_string(b.y) +
                   ", w=" + std::to_string(b.w) + ", h=" + std::to_string(b.h) + ")";
        });

    // --- Serialize / Deserialize BBox ---
    m.def("serialize_bbox", [](const BBox& b) -> std::vector<uint8_t> {
        return serialize(b);
    }, nb::arg("bbox"), "Serialize a BBox to bytes (only possible because BBox is FlatType)");

    m.def("deserialize_bbox", [](const std::vector<uint8_t>& buf) -> BBox {
        return deserialize<BBox>(buf);
    }, nb::arg("buf"), "Deserialize a BBox from bytes");

    // --- Image template demo ---
    // We expose a small image (64x48) for testing since large images would be
    // expensive to copy through Python bindings in this demo context.
    nb::class_<SmallRGB>(m, "ImageRGB")
        .def(nb::init<>())
        .def("width", &SmallRGB::width)
        .def("height", &SmallRGB::height)
        .def("channels", &SmallRGB::channels)
        .def("size", &SmallRGB::size)
        .def("set_pixel", [](SmallRGB& img, int x, int y, int c, uint8_t val) {
            if (x < 0 || x >= img.kWidth || y < 0 || y >= img.kHeight || c < 0 || c >= img.kChannels) {
                throw std::out_of_range("pixel index out of range");
            }
            img.at(x, y, c) = val;
        }, nb::arg("x"), nb::arg("y"), nb::arg("c"), nb::arg("value"))
        .def("get_pixel", [](const SmallRGB& img, int x, int y, int c) -> uint8_t {
            if (x < 0 || x >= img.kWidth || y < 0 || y >= img.kHeight || c < 0 || c >= img.kChannels) {
                throw std::out_of_range("pixel index out of range");
            }
            return img.at(x, y, c);
        }, nb::arg("x"), nb::arg("y"), nb::arg("c"));

    nb::class_<SmallGray>(m, "ImageGray")
        .def(nb::init<>())
        .def("width", &SmallGray::width)
        .def("height", &SmallGray::height)
        .def("channels", &SmallGray::channels)
        .def("size", &SmallGray::size);

    // --- Compile-time size info ---
    m.def("image_rgb_compile_size", []() -> size_t { return SmallRGB::kSize; },
          "Returns the compile-time computed size of ImageRGB (64*48*3)");
    m.def("image_gray_compile_size", []() -> size_t { return SmallGray::kSize; },
          "Returns the compile-time computed size of ImageGray (64*48*1)");
    m.def("hd_rgb_compile_size", []() -> size_t { return HD_RGB::kSize; },
          "Returns the compile-time computed size of HD_RGB (1920*1080*3)");

    // --- Numeric clamp ---
    m.def("clamp_int", [](int val, int lo, int hi) { return clamp_value(val, lo, hi); },
          nb::arg("val"), nb::arg("lo"), nb::arg("hi"));
    m.def("clamp_float", [](float val, float lo, float hi) { return clamp_value(val, lo, hi); },
          nb::arg("val"), nb::arg("lo"), nb::arg("hi"));
}
