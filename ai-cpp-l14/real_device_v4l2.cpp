// The real implementation behind real_device_header.h: V4L2 capture from a
// /dev/video* node. This is the "vendor .so" the mock stands in for -- the
// point of the lesson is that the same tests run against this and the mock.
//
// Kept deliberately minimal (one buffer, one frame, no streaming loop): it
// exists to be a second implementation of the contract, not a capture library.
#include "real_device_header.h"

#include <cerrno>
#include <cstring>
#include <new>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

namespace
{
    // V4L2 ioctls are interrupted by signals far more often than most syscalls;
    // retrying EINTR is required, not defensive.
    int xioctl(int fd, unsigned long request, void *arg)
    {
        int r = 0;
        do
        {
            r = ::ioctl(fd, request, arg);
        } while (r == -1 && errno == EINTR);
        return r;
    }
} // namespace

struct DeviceHandle
{
    int fd = -1;
    void *mapped = nullptr;
    std::size_t mapped_len = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

extern "C" DeviceHandle *device_open(char const *path)
{
    if (path == nullptr) return nullptr;

    int const fd = ::open(path, O_RDWR | O_NONBLOCK, 0);
    if (fd == -1) return nullptr;

    auto *h = new (std::nothrow) DeviceHandle{};
    if (h == nullptr)
    {
        ::close(fd);
        return nullptr;
    }
    h->fd = fd;

    v4l2_capability cap{};
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1 ||
        (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
        (cap.capabilities & V4L2_CAP_STREAMING) == 0)
    {
        device_close(h);
        return nullptr;
    }

    // Let the driver pick its preferred size: the contract does not fix a
    // resolution, and a real camera will not offer the mock's 64x48.
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1)
    {
        device_close(h);
        return nullptr;
    }
    h->width = fmt.fmt.pix.width;
    h->height = fmt.fmt.pix.height;

    v4l2_requestbuffers req{};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1 || req.count < 1)
    {
        device_close(h);
        return nullptr;
    }

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
    {
        device_close(h);
        return nullptr;
    }

    h->mapped_len = buf.length;
    h->mapped = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (h->mapped == MAP_FAILED)
    {
        h->mapped = nullptr;
        device_close(h);
        return nullptr;
    }

    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) == -1)
    {
        device_close(h);
        return nullptr;
    }
    return h;
}

extern "C" int device_read_frame(DeviceHandle *handle, DeviceFrame *out)
{
    if (handle == nullptr || out == nullptr || handle->fd == -1) return -1;

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (xioctl(handle->fd, VIDIOC_QBUF, &buf) == -1) return -1;

    // 5 s: measured 1.6 s to the first frame on a UVC webcam (sensor start-up),
    // 200 ms steady state. A spin count here was a timeout with no time in it.
    constexpr int frame_timeout_ms = 5000;
    for (;;)
    {
        pollfd pfd{.fd = handle->fd, .events = POLLIN, .revents = 0};
        int const ready = ::poll(&pfd, 1, frame_timeout_ms);
        if (ready == -1 && errno == EINTR) continue;
        if (ready <= 0) return -1;

        v4l2_buffer done{};
        done.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        done.memory = V4L2_MEMORY_MMAP;
        if (xioctl(handle->fd, VIDIOC_DQBUF, &done) == 0)
        {
            out->width = handle->width;
            out->height = handle->height;
            out->timestamp_ns = static_cast<std::uint64_t>(done.timestamp.tv_sec) * 1'000'000'000ull +
                                static_cast<std::uint64_t>(done.timestamp.tv_usec) * 1000ull;
            std::size_t const n = sizeof(out->data) < done.bytesused ? sizeof(out->data) : done.bytesused;
            std::memset(out->data, 0, sizeof(out->data));
            std::memcpy(out->data, handle->mapped, n);
            return 0;
        }
        if (errno != EAGAIN) return -1;
    }
}

extern "C" void device_close(DeviceHandle *handle)
{
    if (handle == nullptr) return;
    if (handle->fd != -1)
    {
        auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(handle->fd, VIDIOC_STREAMOFF, &type);
    }
    if (handle->mapped != nullptr) ::munmap(handle->mapped, handle->mapped_len);
    if (handle->fd != -1) ::close(handle->fd);
    delete handle;
}
