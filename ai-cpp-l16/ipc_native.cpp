#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace
{
    std::atomic<int> g_zero_copy_transfers{0};

    [[noreturn]] void raise_errno(char const *what)
    {
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
    }
} // namespace

// ===========================================================================
// Producer-owned POSIX shared memory. Only the producer creates and unlinks
// the segment; a consumer that receives the fd via SCM_RIGHTS maps it
// through ShmFrameView instead (see below) -- it never re-opens by name,
// which is the whole point of passing the fd.
// ===========================================================================
class ShmFrame
{
public:
    ShmFrame(std::string const &name, std::size_t size) : name_(name), size_(size)
    {
        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ < 0)
        {
            raise_errno("shm_open");
        }
        if (ftruncate(fd_, static_cast<off_t>(size_)) < 0)
        {
            close(fd_);
            raise_errno("ftruncate");
        }
        data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED)
        {
            close(fd_);
            raise_errno("mmap");
        }
    }

    ~ShmFrame()
    {
        if (data_ != nullptr && data_ != MAP_FAILED)
        {
            munmap(data_, size_);
        }
        if (fd_ >= 0)
        {
            close(fd_);
        }
        shm_unlink(name_.c_str());
    }

    void write(nb::bytes data)
    {
        if (data.size() != size_)
        {
            throw std::runtime_error("write size does not match frame size");
        }
        std::memcpy(data_, data.c_str(), size_);
    }

    [[nodiscard]] nb::bytes read() const
    {
        return nb::bytes(static_cast<char const *>(data_), size_);
    }

    [[nodiscard]] int fd() const noexcept { return fd_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::string name_;
    std::size_t size_;
    int fd_ = -1;
    void *data_ = nullptr;
};

// A view over an fd received from another process. Never calls shm_open --
// the memory becomes visible purely because the fd (a handle to the same
// kernel-held page cache object) was handed over.
class ShmFrameView
{
public:
    ShmFrameView(int fd, std::size_t size) : fd_(fd), size_(size)
    {
        data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED)
        {
            close(fd_);
            raise_errno("mmap");
        }
    }

    ~ShmFrameView()
    {
        if (data_ != nullptr && data_ != MAP_FAILED)
        {
            munmap(data_, size_);
        }
        if (fd_ >= 0)
        {
            close(fd_);
        }
    }

    [[nodiscard]] nb::bytes read() const
    {
        return nb::bytes(static_cast<char const *>(data_), size_);
    }

private:
    int fd_;
    std::size_t size_;
    void *data_ = nullptr;
};

// ===========================================================================
// Unix domain socket: the control channel. It carries either raw bytes (the
// copy baseline) or an SCM_RIGHTS-passed fd (the zero-copy path) -- never
// both for the same frame, so a benchmark run is unambiguously one or the
// other.
// ===========================================================================
[[nodiscard]] int listen_unix_socket(std::string const &path)
{
    unlink(path.c_str());
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
    {
        raise_errno("socket");
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        close(fd);
        raise_errno("bind");
    }
    if (listen(fd, 1) < 0)
    {
        close(fd);
        raise_errno("listen");
    }
    return fd;
}

// accept()/connect()/send()/recv()/sendmsg()/recvmsg() below all release the
// GIL: each can block waiting on another thread or process, and a blocked
// call that keeps the GIL prevents anyone else -- including a background
// Python thread meant to be driving the other end of this same socket --
// from making progress (see L15).
[[nodiscard]] int accept_unix_socket(int listen_fd)
{
    nb::gil_scoped_release release;
    int client = accept(listen_fd, nullptr, nullptr);
    if (client < 0)
    {
        raise_errno("accept");
    }
    return client;
}

[[nodiscard]] int connect_unix_socket(std::string const &path)
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
    {
        raise_errno("socket");
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    nb::gil_scoped_release release;
    // The server may not have called listen() yet; retry briefly.
    for (int attempt = 0; attempt < 500; ++attempt)
    {
        if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0)
        {
            return fd;
        }
        usleep(10'000);
    }
    close(fd);
    raise_errno("connect");
}

void close_fd(int fd) { close(fd); }

void send_fd(int socket_fd, int fd_to_send)
{
    std::uint8_t tag = 1;
    iovec io{&tag, 1};
    char cmsg_buf[CMSG_SPACE(sizeof(int))] = {};

    msghdr msg{};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

    nb::gil_scoped_release release;
    if (sendmsg(socket_fd, &msg, 0) < 0)
    {
        raise_errno("sendmsg");
    }
}

[[nodiscard]] int recv_fd(int socket_fd)
{
    std::uint8_t tag = 0;
    iovec io{&tag, 1};
    char cmsg_buf[CMSG_SPACE(sizeof(int))] = {};

    msghdr msg{};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    {
        nb::gil_scoped_release release;
        if (recvmsg(socket_fd, &msg, 0) <= 0)
        {
            raise_errno("recvmsg");
        }
    }
    cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == nullptr || cmsg->cmsg_type != SCM_RIGHTS)
    {
        throw std::runtime_error("recvmsg did not carry an SCM_RIGHTS fd");
    }
    int received_fd = 0;
    std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
    g_zero_copy_transfers.fetch_add(1, std::memory_order_relaxed);
    return received_fd;
}

void send_bytes(int socket_fd, nb::bytes data)
{
    // Extract the buffer while the GIL is held (this may be the only
    // reference keeping the underlying Python bytes object alive), then
    // release it for the blocking loop -- a large payload can block on
    // socket buffer space for a while.
    char const *ptr = data.c_str();
    std::size_t const total = data.size();

    nb::gil_scoped_release release;
    std::size_t sent = 0;
    while (sent < total)
    {
        ssize_t written = send(socket_fd, ptr + sent, total - sent, 0);
        if (written <= 0)
        {
            raise_errno("send");
        }
        sent += static_cast<std::size_t>(written);
    }
}

[[nodiscard]] nb::bytes recv_bytes(int socket_fd, std::size_t n)
{
    std::vector<char> buf(n);
    {
        nb::gil_scoped_release release;
        std::size_t received = 0;
        while (received < n)
        {
            ssize_t got = recv(socket_fd, buf.data() + received, n - received, 0);
            if (got <= 0)
            {
                raise_errno("recv");
            }
            received += static_cast<std::size_t>(got);
        }
    }
    return nb::bytes(buf.data(), n);
}

// Verification hook: how many fds this process has actually received via
// SCM_RIGHTS. A silent fallback to the copy path would leave this at 0
// while the benchmark still "succeeds" -- the falsification step from the
// README.
[[nodiscard]] int zero_copy_transfer_count() noexcept
{
    return g_zero_copy_transfers.load(std::memory_order_relaxed);
}

NB_MODULE(ipc_native, m)
{
    nb::class_<ShmFrame>(m, "ShmFrame")
        .def(nb::init<std::string const &, std::size_t>(), nb::arg("name"), nb::arg("size"))
        .def("write", &ShmFrame::write, nb::arg("data"))
        .def("read", &ShmFrame::read)
        .def("fd", &ShmFrame::fd)
        .def("size", &ShmFrame::size);

    nb::class_<ShmFrameView>(m, "ShmFrameView")
        .def(nb::init<int, std::size_t>(), nb::arg("fd"), nb::arg("size"))
        .def("read", &ShmFrameView::read);

    m.def("listen_unix_socket", &listen_unix_socket, nb::arg("path"));
    m.def("accept_unix_socket", &accept_unix_socket, nb::arg("listen_fd"));
    m.def("connect_unix_socket", &connect_unix_socket, nb::arg("path"));
    m.def("close_fd", &close_fd, nb::arg("fd"));
    m.def("send_fd", &send_fd, nb::arg("socket_fd"), nb::arg("fd_to_send"));
    m.def("recv_fd", &recv_fd, nb::arg("socket_fd"));
    m.def("send_bytes", &send_bytes, nb::arg("socket_fd"), nb::arg("data"));
    m.def("recv_bytes", &recv_bytes, nb::arg("socket_fd"), nb::arg("n"));
    m.def("zero_copy_transfer_count", &zero_copy_transfer_count);
}
