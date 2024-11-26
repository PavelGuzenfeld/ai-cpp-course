#include <algorithm>
#include <execution> // for std::execution::unseq
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xsimd/xsimd.hpp>

namespace py = pybind11;

py::array_t<uint8_t> crop_and_resize(
    py::array_t<uint8_t> input_image,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{
    // convert numpy array to cv::mat
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);

    // crop the image
    cv::Rect roi(start_x, start_y, crop_width, crop_height);
    cv::Mat cropped = img(roi);

    // resize the image
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(target_width, target_height));

    // return as numpy array
    return py::array_t<uint8_t>(
        {resized.rows, resized.cols, resized.channels()},
        resized.data);
}

cv::Mat crop_and_resize_with_unseq_imp(
    const cv::Mat &input,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{

    // Step 1: Crop
    cv::Rect roi(start_x, start_y, crop_width, crop_height);
    cv::Mat cropped = input(roi);

    // Step 2: Resize
    cv::Mat resized(target_height, target_width, cropped.type());
    float x_ratio = static_cast<float>(cropped.cols) / target_width;
    float y_ratio = static_cast<float>(cropped.rows) / target_height;

    // Create a range of indices
    std::vector<int> rows(target_height);
    std::iota(rows.begin(), rows.end(), 0); // Fill with 0, 1, ..., target_height - 1

    // Parallelize over rows
    std::for_each(std::execution::unseq, rows.begin(), rows.end(), [&](int y)
                  {
        int src_y = static_cast<int>(y * y_ratio);
        const uchar *src_row = cropped.ptr<uchar>(src_y);
        uchar *dst_row = resized.ptr<uchar>(y);

        // Create a range of columns
        std::vector<int> cols(target_width);
        std::iota(cols.begin(), cols.end(), 0);  // Fill with 0, 1, ..., target_width - 1

        // Parallelize over columns
        std::for_each(std::execution::unseq, cols.begin(), cols.end(), [&](int x) {
            int src_x = static_cast<int>(x * x_ratio);

            // Copy RGB channels
            dst_row[x * 3 + 0] = src_row[src_x * 3 + 0]; // Red
            dst_row[x * 3 + 1] = src_row[src_x * 3 + 1]; // Green
            dst_row[x * 3 + 2] = src_row[src_x * 3 + 2]; // Blue
        }); });

    return resized;
}

// Pybind11 wrapper
py::array_t<uint8_t> crop_and_resize_unseq(
    py::array_t<uint8_t> input_image,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{

    // Convert NumPy array to cv::Mat
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);

    // Apply SIMD crop and resize
    cv::Mat result = crop_and_resize_with_unseq_imp(
        img, start_x, start_y, crop_width, crop_height, target_width, target_height);

    // Return as NumPy array
    return py::array_t<uint8_t>(
        {result.rows, result.cols, result.channels()}, // Shape
        result.data                                    // Data
    );
}
cv::Mat crop_and_resize_with_xsimd_imp(
    const cv::Mat &input,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{
    // Ensure the ROI is within bounds
    cv::Rect roi(start_x, start_y, crop_width, crop_height);
    roi &= cv::Rect(0, 0, input.cols, input.rows);

    // Step 1: Crop directly into a new Mat
    cv::Mat cropped(input, roi);

    // Step 2: Initialize the resized image output
    cv::Mat resized(target_height, target_width, input.type());
    float x_ratio = static_cast<float>(cropped.cols) / target_width;
    float y_ratio = static_cast<float>(cropped.rows) / target_height;

    // Process rows using SIMD
    for (int y = 0; y < target_height; ++y)
    {
        int src_y = static_cast<int>(y * y_ratio);
        const uchar *src_row = cropped.ptr<uchar>(src_y);
        uchar *dst_row = resized.ptr<uchar>(y);

        // Use SIMD to process pixels in batches
        int x = 0;
        for (; x <= target_width - xsimd::batch<uint8_t>::size; x += xsimd::batch<uint8_t>::size)
        {
            int src_x_start = static_cast<int>(x * x_ratio) * 3;

            // Load and process RGB channels in SIMD batches
            xsimd::batch<uint8_t> r = xsimd::load_unaligned(&src_row[src_x_start + 0]);
            xsimd::batch<uint8_t> g = xsimd::load_unaligned(&src_row[src_x_start + 1]);
            xsimd::batch<uint8_t> b = xsimd::load_unaligned(&src_row[src_x_start + 2]);

            r.store_unaligned(&dst_row[x * 3 + 0]);
            g.store_unaligned(&dst_row[x * 3 + 1]);
            b.store_unaligned(&dst_row[x * 3 + 2]);
        }

        // Process remaining pixels (scalar fallback)
        for (; x < target_width; ++x)
        {
            int src_x = static_cast<int>(x * x_ratio);
            dst_row[x * 3 + 0] = src_row[src_x * 3 + 0];
            dst_row[x * 3 + 1] = src_row[src_x * 3 + 1];
            dst_row[x * 3 + 2] = src_row[src_x * 3 + 2];
        }
    }

    return resized;
}

// Pybind11 wrapper
py::array_t<uint8_t> crop_and_resize_xsimd(
    py::array_t<uint8_t> input_image,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{
    // Convert NumPy array directly to cv::Mat without copying
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);

    // Apply SIMD crop and resize
    cv::Mat result = crop_and_resize_with_xsimd_imp(
        img, start_x, start_y, crop_width, crop_height, target_width, target_height);

    // Return the resized image as a NumPy array
    return py::array_t<uint8_t>(
        {result.rows, result.cols, result.channels()}, // Shape
        {static_cast<size_t>(result.step[0]),          // Row stride
         static_cast<size_t>(result.step[1]),          // Column stride
         static_cast<size_t>(1)},                      // Channel stride
        result.data                                    // Data pointer
    );
}

void crop_and_resize_with_xsimd_inplace(
    cv::Mat &input,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{
    // Ensure the ROI is within bounds
    cv::Rect roi(start_x, start_y, crop_width, crop_height);
    roi &= cv::Rect(0, 0, input.cols, input.rows);

    // Step 1: Adjust the ROI (Crop in-place)
    cv::Mat cropped(input, roi);

    // Step 2: Resize in-place
    input = cropped;                                           // Update input to the cropped region
    input.create(target_height, target_width, cropped.type()); // Resize in-place

    float x_ratio = static_cast<float>(cropped.cols) / target_width;
    float y_ratio = static_cast<float>(cropped.rows) / target_height;

    for (int y = 0; y < target_height; ++y)
    {
        int src_y = static_cast<int>(y * y_ratio);
        const uchar *src_row = cropped.ptr<uchar>(src_y);
        uchar *dst_row = input.ptr<uchar>(y);

        // Use SIMD to process pixels in batches
        int x = 0;
        for (; x <= target_width - xsimd::batch<uint8_t>::size; x += xsimd::batch<uint8_t>::size)
        {
            int src_x_start = static_cast<int>(x * x_ratio) * 3;

            // Load and process RGB channels in SIMD batches
            xsimd::batch<uint8_t> r = xsimd::load_unaligned(&src_row[src_x_start + 0]);
            xsimd::batch<uint8_t> g = xsimd::load_unaligned(&src_row[src_x_start + 1]);
            xsimd::batch<uint8_t> b = xsimd::load_unaligned(&src_row[src_x_start + 2]);

            r.store_unaligned(&dst_row[x * 3 + 0]);
            g.store_unaligned(&dst_row[x * 3 + 1]);
            b.store_unaligned(&dst_row[x * 3 + 2]);
        }

        // Process remaining pixels (scalar fallback)
        for (; x < target_width; ++x)
        {
            int src_x = static_cast<int>(x * x_ratio);
            dst_row[x * 3 + 0] = src_row[src_x * 3 + 0];
            dst_row[x * 3 + 1] = src_row[src_x * 3 + 1];
            dst_row[x * 3 + 2] = src_row[src_x * 3 + 2];
        }
    }
}

void crop_and_resize_xsimd_inplace(
    py::array_t<uint8_t> input_image,
    int start_x, int start_y,
    int crop_width, int crop_height,
    int target_width, int target_height)
{
    // Convert NumPy array directly to cv::Mat without copying
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);

    // Perform in-place crop and resize
    crop_and_resize_with_xsimd_inplace(img, start_x, start_y, crop_width, crop_height, target_width, target_height);
}

PYBIND11_MODULE(cpp_image_processor, m)
{
    m.doc() = "Crop and resize image functions using xsimd for SIMD acceleration";

    m.def("crop_and_resize", &crop_and_resize, "Crop and resize image",
          py::arg("input_image"),
          py::arg("start_x"), py::arg("start_y"),
          py::arg("crop_width"), py::arg("crop_height"),
          py::arg("target_width"), py::arg("target_height"));

    m.def("crop_and_resize_unseq", &crop_and_resize_unseq, "Crop and resize image with SIMD using std::execution::unseq",
          py::arg("input_image"), py::arg("start_x"), py::arg("start_y"),
          py::arg("crop_width"), py::arg("crop_height"),
          py::arg("target_width"), py::arg("target_height"));

    m.def("crop_and_resize_xsimd", &crop_and_resize_xsimd, "Crop and resize image with SIMD using xsimd",
          py::arg("input_image"), py::arg("start_x"), py::arg("start_y"),
          py::arg("crop_width"), py::arg("crop_height"),
          py::arg("target_width"), py::arg("target_height"));

    m.def("crop_and_resize_xsimd_inplace", &crop_and_resize_xsimd_inplace, "Crop and resize image in-place with SIMD using xsimd",
            py::arg("input_image"), py::arg("start_x"), py::arg("start_y"),
            py::arg("crop_width"), py::arg("crop_height"),
            py::arg("target_width"), py::arg("target_height"));
}