// Step 1.8 of the machine model: the fixed cost of crossing a boundary.
// Every number here is a *per-crossing* cost, in nanoseconds, independent of
// how much work sits on either side. That is what makes it a tax.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <vector>

#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace nb = nanobind;

namespace
{
    using Clock = std::chrono::steady_clock;

    double ns_per_op(Clock::time_point t0, std::size_t ops)
    {
        double const ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        return ns / static_cast<double>(ops);
    }

    // getpid is the cheapest real syscall: no arguments, no work in the
    // kernel beyond the transition itself. Measured through syscall() so
    // glibc's caching cannot turn it into a load.
    double syscall_floor_ns(std::size_t n)
    {
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            long const r = ::syscall(SYS_getpid);
            asm volatile("" : : "r"(r) : "memory");
        }
        return ns_per_op(t0, n);
    }

    // The same clock read, once through the vDSO and once forced into a real
    // syscall. The gap is what the vDSO buys, and it is why a timer in a hot
    // loop is not automatically a syscall.
    double clock_gettime_vdso_ns(std::size_t n)
    {
        timespec ts{};
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            ::clock_gettime(CLOCK_MONOTONIC, &ts);
            asm volatile("" : : "r"(ts.tv_nsec) : "memory");
        }
        return ns_per_op(t0, n);
    }

    double clock_gettime_syscall_ns(std::size_t n)
    {
        timespec ts{};
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            ::syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &ts);
            asm volatile("" : : "r"(ts.tv_nsec) : "memory");
        }
        return ns_per_op(t0, n);
    }

    // Allocation that the allocator can satisfy from its free list, versus
    // allocation large enough to hit mmap and fault pages in. The second is
    // the one that shows up as a mysterious per-frame cost.
    double malloc_small_ns(std::size_t n)
    {
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            void *p = std::malloc(64);
            asm volatile("" : : "r"(p) : "memory");
            std::free(p);
        }
        return ns_per_op(t0, n);
    }

    double malloc_page_faulted_ns(std::size_t n, std::size_t bytes)
    {
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            auto *p = static_cast<unsigned char *>(std::malloc(bytes));
            for (std::size_t off = 0; off < bytes; off += 4096) p[off] = 1; // fault it in
            asm volatile("" : : "r"(p) : "memory");
            std::free(p);
        }
        return ns_per_op(t0, n);
    }

    // Uncontended lock/unlock. Contended is a different measurement and a
    // different number; this is the floor you pay even when nobody is there.
    double mutex_uncontended_ns(std::size_t n)
    {
        std::mutex m;
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            m.lock();
            asm volatile("" : : : "memory");
            m.unlock();
        }
        return ns_per_op(t0, n);
    }

    // One byte out and back through a pipe and through a socketpair. Both
    // are two syscalls plus a scheduler wake; the point is that they are not
    // free and that shared memory does not pay this at all.
    double roundtrip_ns(int const fds[2], std::size_t n)
    {
        char c = 'x';
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
        {
            if (::write(fds[1], &c, 1) != 1) return -1.0;
            if (::read(fds[0], &c, 1) != 1) return -1.0;
        }
        return ns_per_op(t0, n);
    }

    double pipe_roundtrip_ns(std::size_t n)
    {
        int fds[2];
        if (::pipe(fds) != 0) return -1.0;
        double const r = roundtrip_ns(fds, n);
        ::close(fds[0]);
        ::close(fds[1]);
        return r;
    }

    double socketpair_roundtrip_ns(std::size_t n)
    {
        int fds[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) return -1.0;
        double const r = roundtrip_ns(fds, n);
        ::close(fds[0]);
        ::close(fds[1]);
        return r;
    }

    // The FFI crossing, measured two ways. noop() is called once per item
    // from Python; noop_batch(n) is called once for n items. The difference
    // between their per-item costs is the crossing, and it is the whole
    // argument for batching at the boundary.
    void noop() {}

    std::uint64_t noop_batch(std::size_t n)
    {
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) acc += i;
        return acc;
    }

    // Same signature, real work inside, so a reader can see the crossing
    // stop mattering once the body is big enough.
    std::uint64_t work_per_call(std::size_t inner)
    {
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < inner; ++i) acc += i * 2654435761u;
        return acc;
    }
} // namespace

NB_MODULE(tax_bench, m)
{
    m.doc() = "Per-crossing costs for the machine model (step 1.8)";
    m.def("syscall_floor_ns", &syscall_floor_ns, nb::arg("n") = 200000);
    m.def("clock_gettime_vdso_ns", &clock_gettime_vdso_ns, nb::arg("n") = 200000);
    m.def("clock_gettime_syscall_ns", &clock_gettime_syscall_ns, nb::arg("n") = 200000);
    m.def("malloc_small_ns", &malloc_small_ns, nb::arg("n") = 200000);
    m.def("malloc_page_faulted_ns", &malloc_page_faulted_ns,
          nb::arg("n") = 2000, nb::arg("bytes") = 1u << 20);
    m.def("mutex_uncontended_ns", &mutex_uncontended_ns, nb::arg("n") = 200000);
    m.def("pipe_roundtrip_ns", &pipe_roundtrip_ns, nb::arg("n") = 20000);
    m.def("socketpair_roundtrip_ns", &socketpair_roundtrip_ns, nb::arg("n") = 20000);
    m.def("noop", &noop);
    m.def("noop_batch", &noop_batch, nb::arg("n"));
    m.def("work_per_call", &work_per_call, nb::arg("inner"));
}
