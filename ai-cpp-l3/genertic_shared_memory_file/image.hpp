#pragma once

#include <array>   // std::array
#include <cstddef> // std::size_t
#include <cstdint> // std::uint8_t and std::uint64_t

namespace img
{

    enum class ImageType : std::uint8_t
    {
        RGB,
        RGBA,
        NV12,
    };

    constexpr float channels(ImageType type)
    {
        switch (type)
        {
        case ImageType::RGB:
            return 3.0;
        case ImageType::RGBA:
            return 4.0;
        case ImageType::NV12:
            return 1.5; // Approximated as integer (NV12 uses 1.5 bytes per
                        // pixel)
        default:
            return 3; // Default to RGB channels
        }
    }

    constexpr size_t CACHE_LINE_SIZE = 64;
#pragma pack(push, 1)
    template <std::size_t WIDTH, std::size_t HEIGHT, ImageType TYPE>
    struct Image
    {
        static std::size_t const width = WIDTH;
        static std::size_t const height = HEIGHT;
        static ImageType const type = TYPE;
        static constexpr std::size_t size = WIDTH * HEIGHT * channels(TYPE);

        alignas(CACHE_LINE_SIZE) uint64_t timestamp;
        alignas(CACHE_LINE_SIZE) uint64_t frame_number;
        alignas(CACHE_LINE_SIZE) std::array<std::uint8_t, size> data;
    };
#pragma pack(pop)

    // Typedefs for common image types
    using ImageFHD_RGB = Image<1920, 1080, ImageType::RGB>;
    using ImageFHD_RGBA = Image<1920, 1080, ImageType::RGBA>;
    using ImageFHD_NV12 = Image<1920, 1080, ImageType::NV12>;
    using Image4K_RGB = Image<3840, 2160, ImageType::RGB>;
    using Image4K_RGBA = Image<3840, 2160, ImageType::RGBA>;
    using Image4K_NV12 = Image<3840, 2160, ImageType::NV12>;

    // Static assertions to verify sizes
    static_assert(sizeof(ImageFHD_RGB) == (1920 * 1080 * 3 + 2 * sizeof(uint64_t)));
    static_assert(sizeof(ImageFHD_RGBA) == (1920 * 1080 * 4 + 2 * sizeof(uint64_t)));
    static_assert(sizeof(ImageFHD_NV12) == (1920 * 1080 * 1.5 + 2 * sizeof(uint64_t)));
    static_assert(sizeof(Image4K_RGB) == (3840 * 2160 * 3 + 2 * sizeof(uint64_t)));
    static_assert(sizeof(Image4K_RGBA) == (3840 * 2160 * 4 + 2 * sizeof(uint64_t)));
    static_assert(sizeof(Image4K_NV12) == (3840 * 2160 * 1.5 + 2 * sizeof(uint64_t)));

} // namespace img
