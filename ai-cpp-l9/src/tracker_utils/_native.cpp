#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nb = nanobind;

struct BBox
{
    double x, y, w, h;

    BBox(double x, double y, double w, double h) : x{x}, y{y}, w{w}, h{h}
    {
        if (w < 0.0 || h < 0.0)
        {
            throw std::invalid_argument("width and height must be non-negative");
        }
    }

    [[nodiscard]] double cx() const noexcept { return x + w / 2.0; }
    [[nodiscard]] double cy() const noexcept { return y + h / 2.0; }
    [[nodiscard]] double area() const noexcept { return w * h; }

    [[nodiscard]] double aspect_ratio() const noexcept
    {
        return (h > 0.0) ? w / h : 0.0;
    }

    [[nodiscard]] double iou(const BBox& other) const noexcept
    {
        double x1 = std::max(x, other.x);
        double y1 = std::max(y, other.y);
        double x2 = std::min(x + w, other.x + other.w);
        double y2 = std::min(y + h, other.y + other.h);

        double inter_w = std::max(0.0, x2 - x1);
        double inter_h = std::max(0.0, y2 - y1);
        double inter_area = inter_w * inter_h;

        double union_area = area() + other.area() - inter_area;
        return (union_area > 0.0) ? inter_area / union_area : 0.0;
    }

    [[nodiscard]] bool contains_point(double px, double py) const noexcept
    {
        return px >= x && px <= x + w && py >= y && py <= y + h;
    }

    // Return bbox as a numpy array [x, y, w, h] — zero-copy via ndarray
    [[nodiscard]] nb::ndarray<nb::numpy, double, nb::shape<4>> to_array()
    {
        auto* data = new double[4]{x, y, w, h};
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::shape<4>>(data, {4}, owner);
    }

    // Construct a BBox from a numpy array of 4 doubles
    static BBox from_array(nb::ndarray<double, nb::shape<4>> arr)
    {
        const double* ptr = arr.data();
        return BBox{ptr[0], ptr[1], ptr[2], ptr[3]};
    }
};

// Module name matches the import path: tracker_utils._native
NB_MODULE(_native, m)
{
    m.doc() = "C++ BoundingBox with nanobind — packaged as tracker_utils._native";

    nb::class_<BBox>(m, "BBox")
        .def(nb::init<double, double, double, double>(),
             nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("h"))
        .def_rw("x", &BBox::x)
        .def_rw("y", &BBox::y)
        .def_rw("w", &BBox::w)
        .def_rw("h", &BBox::h)
        .def_prop_ro("cx", &BBox::cx)
        .def_prop_ro("cy", &BBox::cy)
        .def_prop_ro("area", &BBox::area)
        .def_prop_ro("aspect_ratio", &BBox::aspect_ratio)
        .def("iou", &BBox::iou, nb::arg("other"))
        .def("contains_point", &BBox::contains_point, nb::arg("px"), nb::arg("py"))
        .def("to_array", &BBox::to_array)
        .def_static("from_array", &BBox::from_array, nb::arg("arr"))
        .def("__repr__", [](const BBox& b) {
            return "BBox(x=" + std::to_string(b.x) + ", y=" + std::to_string(b.y) +
                   ", w=" + std::to_string(b.w) + ", h=" + std::to_string(b.h) + ")";
        });
}
