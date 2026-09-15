/**
 * Python-facing wrapper around stride_view.hpp's correct and sheared copy
 * paths, for exact byte-level testing without needing OpenCV or a
 * sanitizer build in the loop. The overrun path is deliberately not bound
 * here -- it is exercised by the standalone stride_demo executable under
 * ASan instead (see CMakeLists.txt / ci.yml), not through pybind11.
 */
#include "stride_view.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;

namespace
{

stride_lesson::View make_view(const py::array_t<std::uint8_t> &padded_src, int width, int height,
                               int channels, std::size_t stride)
{
    auto buf = padded_src.request();
    if (static_cast<std::size_t>(buf.size) < stride * static_cast<std::size_t>(height))
    {
        throw std::invalid_argument("padded_src is smaller than stride * height");
    }
    return stride_lesson::View{static_cast<const std::uint8_t *>(buf.ptr), width, height, channels, stride};
}

// The single-ssize_t array_t constructor (`array_t(ssize_t count, ...)`)
// produces a broadcast-strided (not freshly allocated) array under the
// apt-packaged pybind11 2.9.1 this project builds against -- every element
// silently aliases element 0, so a memcpy into it "succeeds" while writing
// the same byte repeatedly. The explicit shape-vector constructor goes
// through a different, correct code path on both 2.9.1 and modern pybind11.
py::array_t<std::uint8_t> make_output(std::size_t n)
{
    return py::array_t<std::uint8_t>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
}

py::array_t<std::uint8_t> copy_using_stride(py::array_t<std::uint8_t> padded_src, int width, int height,
                                             int channels, std::size_t stride)
{
    stride_lesson::View src = make_view(padded_src, width, height, channels, stride);
    py::array_t<std::uint8_t> out = make_output(src.row_bytes() * static_cast<std::size_t>(height));
    stride_lesson::copy_using_stride(src, out.mutable_data());
    return out;
}

py::array_t<std::uint8_t> copy_using_width_as_stride(py::array_t<std::uint8_t> padded_src, int width, int height,
                                                      int channels, std::size_t stride)
{
    stride_lesson::View src = make_view(padded_src, width, height, channels, stride);
    py::array_t<std::uint8_t> out = make_output(src.row_bytes() * static_cast<std::size_t>(height));
    stride_lesson::copy_using_width_as_stride(src, out.mutable_data());
    return out;
}

} // namespace

PYBIND11_MODULE(stride_view_native, m)
{
    m.doc() = "Stride-aware vs width-as-stride image copy (capstone-adjacent, L2 stride lesson)";
    m.def("copy_using_stride", &copy_using_stride,
          py::arg("padded_src"), py::arg("width"), py::arg("height"), py::arg("channels"), py::arg("stride"));
    m.def("copy_using_width_as_stride", &copy_using_width_as_stride,
          py::arg("padded_src"), py::arg("width"), py::arg("height"), py::arg("channels"), py::arg("stride"));
}
