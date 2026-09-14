#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace safe_shm::img
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
            return 3.0f;
        case ImageType::RGBA:
            return 4.0f;
        case ImageType::NV12:
            return 1.5f;
        }
        return 3.0f;
    }

#pragma pack(push, 1)
    template <std::size_t WIDTH, std::size_t HEIGHT, ImageType TYPE>
    struct Image
    {
        static constexpr std::size_t width = WIDTH;
        static constexpr std::size_t height = HEIGHT;
        static constexpr ImageType type = TYPE;
        static constexpr std::size_t size =
            static_cast<std::size_t>(WIDTH * HEIGHT * channels(TYPE));

        std::uint64_t timestamp;
        std::uint64_t frame_number;
        std::array<std::uint8_t, size> data;
    };
#pragma pack(pop)

    using ImageFHD_RGB = Image<1920, 1080, ImageType::RGB>;
    using ImageFHD_RGBA = Image<1920, 1080, ImageType::RGBA>;
    using ImageFHD_NV12 = Image<1920, 1080, ImageType::NV12>;
    using Image4K_RGB = Image<3840, 2160, ImageType::RGB>;
    using Image4K_RGBA = Image<3840, 2160, ImageType::RGBA>;
    using Image4K_NV12 = Image<3840, 2160, ImageType::NV12>;

    static_assert(sizeof(ImageFHD_RGB) == 1920 * 1080 * 3 + 2 * sizeof(std::uint64_t));
    static_assert(sizeof(ImageFHD_RGBA) == 1920 * 1080 * 4 + 2 * sizeof(std::uint64_t));
    static_assert(sizeof(Image4K_RGB) == 3840 * 2160 * 3 + 2 * sizeof(std::uint64_t));
    static_assert(sizeof(Image4K_RGBA) == 3840 * 2160 * 4 + 2 * sizeof(std::uint64_t));

} // namespace safe_shm::img
