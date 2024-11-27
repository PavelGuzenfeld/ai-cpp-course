#include <algorithm>
#include <execution>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Helper function for scalar row processing
void process_row(const uchar *src_row, uchar *dst_row, int src_cols, float x_ratio, int dst_cols)
{
    for (int x = 0; x < dst_cols; ++x)
    {
        int src_x = static_cast<int>(x * x_ratio);
        dst_row[x * 3 + 0] = src_row[src_x * 3 + 0]; // red
        dst_row[x * 3 + 1] = src_row[src_x * 3 + 1]; // green
        dst_row[x * 3 + 2] = src_row[src_x * 3 + 2]; // blue
    }
}

// Generic crop and resize implementation
cv::Mat crop_and_resize_generic(
    const cv::Mat &input,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height,
    const std::string &mode)
{
    // Step 1: Crop the image
    cv::Rect roi(start_x, start_y, crop_width, crop_height);
    cv::Mat cropped(input, roi);

    // Step 2: Resize the image
    cv::Mat resized(target_height, target_width, cropped.type());
    float x_ratio = static_cast<float>(cropped.cols) / target_width;
    float y_ratio = static_cast<float>(cropped.rows) / target_height;

    if (mode == "unseq")
    {
        // Use std::execution::unseq for parallel row processing
        std::vector<int> rows(target_height);
        std::iota(rows.begin(), rows.end(), 0);

        std::for_each(std::execution::unseq, rows.begin(), rows.end(), [&](int y)
                      {
            int src_y = static_cast<int>(y * y_ratio);
            process_row(cropped.ptr<uchar>(src_y), resized.ptr<uchar>(y), cropped.cols, x_ratio, target_width); });
    }
    else if (mode == "par")
    {
        // Use std::execution::par for parallel row processing
        std::vector<int> rows(target_height);
        std::iota(rows.begin(), rows.end(), 0);

        std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int y)
                      {
            int src_y = static_cast<int>(y * y_ratio);
            process_row(cropped.ptr<uchar>(src_y), resized.ptr<uchar>(y), cropped.cols, x_ratio, target_width); });
    }
    else
    {
        // Default scalar row processing
        for (int y = 0; y < target_height; ++y)
        {
            int src_y = static_cast<int>(y * y_ratio);
            process_row(cropped.ptr<uchar>(src_y), resized.ptr<uchar>(y), cropped.cols, x_ratio, target_width);
        }
    }

    return resized;
}

// Pybind11 wrapper for crop and resize
py::array_t<uint8_t> crop_and_resize(
    py::array_t<uint8_t> input_image,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height,
    const std::string &mode = "scalar")
{
    // Convert NumPy array to cv::Mat
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);

    // Perform crop and resize
    cv::Mat result = crop_and_resize_generic(
        img, start_x, start_y, crop_width, crop_height, target_width, target_height, mode);

    // Convert back to NumPy array
    return py::array_t<uint8_t>(
        {result.rows, result.cols, result.channels()}, // Shape
        {static_cast<size_t>(result.step[0]),          // Row stride
         static_cast<size_t>(result.step[1]),          // Column stride
         static_cast<size_t>(1)},                      // Channel stride
        result.data                                    // Data pointer
    );
}

// Pybind11 module
PYBIND11_MODULE(cpp_image_processor, m)
{
    m.doc() = "Data-driven crop and resize module";

    m.def("crop_and_resize", &crop_and_resize,
          "Crop and resize image with data-driven mode selection",
          py::arg("input_image"),
          py::arg("start_x"), py::arg("start_y"),
          py::arg("crop_width"), py::arg("crop_height"),
          py::arg("target_width"), py::arg("target_height"),
          py::arg("mode") = "scalar");
}
