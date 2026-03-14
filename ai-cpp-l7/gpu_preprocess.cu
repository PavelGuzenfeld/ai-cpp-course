/**
 * Fused CUDA preprocessing kernel.
 *
 * Replaces the tracker_engine TRT_Preprocessor.process() pattern of:
 *   image.astype(float32) → normalize → transpose HWC→CHW → copy to GPU
 *
 * This single kernel takes uint8 HWC input on GPU and produces float32 CHW
 * normalized output on GPU — zero CPU involvement after the initial transfer.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

// ─── Nanobind includes ──────────────────────────────────────────────────────
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// ─── CUDA error checking ────────────────────────────────────────────────────

#ifdef HAVE_CUDA

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

// ─── Fused preprocess kernel ─────────────────────────────────────────────────

/**
 * Single kernel: uint8 HWC → float32 CHW + normalize.
 *
 * Each thread handles one element (one channel of one pixel).
 * Reads from HWC layout, writes to CHW layout.
 */
__global__ void fused_preprocess_kernel(
    const uint8_t* __restrict__ input,   // [H, W, C] uint8
    float* __restrict__ output,          // [C, H, W] float32
    int height, int width, int channels,
    const float* __restrict__ mean,      // [C]
    const float* __restrict__ std)       // [C]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width * channels;
    if (idx >= total) return;

    // Decompose flat index into (h, w, c) for HWC layout
    int c = idx % channels;
    int w = (idx / channels) % width;
    int h = idx / (channels * width);

    // Read uint8 from HWC position
    float pixel = static_cast<float>(input[h * width * channels + w * channels + c]);

    // Normalize: scale to [0,1] then apply mean/std
    pixel = (pixel / 255.0f - mean[c]) / std[c];

    // Write to CHW position
    output[c * height * width + h * width + w] = pixel;
}

/**
 * Host-side wrapper for the fused preprocess kernel.
 *
 * Takes a numpy uint8 HWC image and returns a numpy float32 CHW array.
 * All GPU allocation and transfer is handled internally.
 */
nb::ndarray<nb::numpy, float> fused_preprocess(
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<3>> input,
    std::vector<float> mean,
    std::vector<float> std_dev)
{
    int height = input.shape(0);
    int width = input.shape(1);
    int channels = input.shape(2);
    int total = height * width * channels;

    if (mean.size() != static_cast<size_t>(channels) ||
        std_dev.size() != static_cast<size_t>(channels)) {
        throw std::invalid_argument("mean and std must have length == channels");
    }

    // Allocate device memory
    uint8_t* d_input = nullptr;
    float* d_output = nullptr;
    float* d_mean = nullptr;
    float* d_std = nullptr;

    size_t input_bytes = total * sizeof(uint8_t);
    size_t output_bytes = total * sizeof(float);
    size_t param_bytes = channels * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean, param_bytes));
    CUDA_CHECK(cudaMalloc(&d_std, param_bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, mean.data(), param_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, std_dev.data(), param_bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;
    fused_preprocess_kernel<<<blocks, threads_per_block>>>(
        d_input, d_output, height, width, channels, d_mean, d_std);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    float* host_output = new float[total];
    CUDA_CHECK(cudaMemcpy(host_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_std);

    // Return as numpy array with CHW shape; nanobind takes ownership via deleter
    size_t shape[3] = {
        static_cast<size_t>(channels),
        static_cast<size_t>(height),
        static_cast<size_t>(width)
    };
    nb::capsule owner(host_output, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float>(host_output, 3, shape, owner);
}

/**
 * Check if CUDA is available at runtime.
 */
bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

#else  // !HAVE_CUDA — CPU fallback with same interface

#include <algorithm>

nb::ndarray<nb::numpy, float> fused_preprocess(
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<3>> input,
    std::vector<float> mean,
    std::vector<float> std_dev)
{
    int height = input.shape(0);
    int width = input.shape(1);
    int channels = input.shape(2);
    int total = height * width * channels;

    if (mean.size() != static_cast<size_t>(channels) ||
        std_dev.size() != static_cast<size_t>(channels)) {
        throw std::invalid_argument("mean and std must have length == channels");
    }

    float* host_output = new float[total];
    const uint8_t* src = input.data();

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                float pixel = static_cast<float>(src[h * width * channels + w * channels + c]);
                pixel = (pixel / 255.0f - mean[c]) / std_dev[c];
                host_output[c * height * width + h * width + w] = pixel;
            }
        }
    }

    size_t shape[3] = {
        static_cast<size_t>(channels),
        static_cast<size_t>(height),
        static_cast<size_t>(width)
    };
    nb::capsule owner(host_output, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float>(host_output, 3, shape, owner);
}

bool cuda_available() { return false; }

#endif  // HAVE_CUDA


// ─── Nanobind module definition ──────────────────────────────────────────────

#ifndef MODULE_NAME
#define MODULE_NAME gpu_preprocess
#endif

NB_MODULE(MODULE_NAME, m) {
    m.doc() = "Fused GPU preprocessing: uint8 HWC → float32 CHW + normalize";

    m.def("fused_preprocess", &fused_preprocess,
          nb::arg("input"), nb::arg("mean"), nb::arg("std"),
          "Fused preprocess: uint8 HWC image → float32 CHW normalized.\n"
          "mean and std are per-channel (e.g., ImageNet [0.485, 0.456, 0.406]).");

    m.def("cuda_available", &cuda_available,
          "Returns True if a CUDA-capable GPU is available.");
}
