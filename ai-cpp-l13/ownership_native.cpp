#include <nanobind/nanobind.h>

#include <cstdint>
#include <unordered_set>
#include <utility>

namespace nb = nanobind;

// ===========================================================================
// A minimal C API stand-in: `handle_t create(); void destroy(handle_t);`
//
// Tracks live handles and double-destroys so tests can observe a leak or a
// double-free directly, instead of relying on a sanitizer to catch it.
// ===========================================================================
namespace c_api
{
    using handle_t = std::uint64_t;

    namespace
    {
        std::uint64_t next_id = 1;
        std::unordered_set<handle_t> live_handles;
        int double_destroy_count_ = 0;
    }

    [[nodiscard]] handle_t create()
    {
        handle_t h = next_id++;
        live_handles.insert(h);
        return h;
    }

    void destroy(handle_t h) noexcept
    {
        if (live_handles.erase(h) == 0)
        {
            // The handle was already destroyed (or never created) — this is
            // the double-free the RAII wrapper below exists to prevent.
            ++double_destroy_count_;
        }
    }

    [[nodiscard]] std::size_t live_count() noexcept { return live_handles.size(); }
    [[nodiscard]] int double_destroy_count() noexcept { return double_destroy_count_; }

    void reset_for_test() noexcept
    {
        live_handles.clear();
        double_destroy_count_ = 0;
    }
} // namespace c_api

// ===========================================================================
// OwnedHandle: move-only RAII wrapper over c_api::handle_t.
//
// The teachable point is not "wrap it in RAII" — it's that an *owning*
// wrapper applied to a *borrowed* handle is a double-free. `release()` is
// the escape hatch: it hands ownership back out without destroying.
// ===========================================================================
class OwnedHandle
{
public:
    OwnedHandle() : handle_(c_api::create()), owns_(true) {}

    // Wrap an existing handle. `takes_ownership` must be true only when this
    // wrapper is the sole owner — passing a borrowed handle with
    // takes_ownership=true reproduces the production bug in PRODUCTION_PLAN.md
    // §1.1: the destructor frees memory owned by someone else.
    explicit OwnedHandle(c_api::handle_t h, bool takes_ownership)
        : handle_(h), owns_(takes_ownership)
    {
    }

    OwnedHandle(OwnedHandle const &) = delete;
    OwnedHandle &operator=(OwnedHandle const &) = delete;

    OwnedHandle(OwnedHandle &&other) noexcept
        : handle_(other.handle_), owns_(other.owns_)
    {
        other.owns_ = false;
    }

    OwnedHandle &operator=(OwnedHandle &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle_ = other.handle_;
            owns_ = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }

    ~OwnedHandle() { reset(); }

    [[nodiscard]] c_api::handle_t get() const noexcept { return handle_; }
    [[nodiscard]] bool owns() const noexcept { return owns_; }

    // Hand ownership back to the caller. After this call the wrapper is
    // non-owning and its destructor will not touch the handle.
    [[nodiscard]] c_api::handle_t release() noexcept
    {
        owns_ = false;
        return handle_;
    }

private:
    void reset() noexcept
    {
        if (owns_)
        {
            c_api::destroy(handle_);
            owns_ = false;
        }
    }

    c_api::handle_t handle_ = 0;
    bool owns_ = false;
};

// Exercised from Python to prove the C++ move constructor steals the handle
// and nulls the source's ownership — a Python-level `b = a` cannot exercise
// this, since it aliases the same wrapped object instead of moving it.
[[nodiscard]] bool move_ctor_steals_and_nulls_source()
{
    c_api::reset_for_test();
    OwnedHandle a;
    auto const original = a.get();
    OwnedHandle b(std::move(a));
    return b.owns() && !a.owns() && b.get() == original;
}

[[nodiscard]] bool move_assign_steals_and_nulls_source()
{
    c_api::reset_for_test();
    OwnedHandle a;
    OwnedHandle b;
    auto const a_handle = a.get();
    b = std::move(a);
    return b.owns() && !a.owns() && b.get() == a_handle;
}

// A function that *borrows* a handle it does not own, for use in a function
// signature — the correct counterpart to the buggy wrap below.
void use_borrowed(c_api::handle_t /*h*/) noexcept
{
    // A real component would read through the handle here without owning it.
}

// The bug from PRODUCTION_PLAN.md §1.1, reproduced on purpose: wraps a
// borrowed handle as if this wrapper owned it. When both this wrapper and
// the real owner are destroyed, `destroy()` is called twice on the same id.
[[nodiscard]] OwnedHandle wrap_borrowed_incorrectly(c_api::handle_t h)
{
    return OwnedHandle(h, /*takes_ownership=*/true); // BUG: does not own `h`
}

// The fix: wrap a borrowed handle as non-owning. Its destructor is a no-op.
[[nodiscard]] OwnedHandle wrap_borrowed_correctly(c_api::handle_t h)
{
    return OwnedHandle(h, /*takes_ownership=*/false);
}

NB_MODULE(ownership_native, m)
{
    m.def("reset_for_test", &c_api::reset_for_test);
    m.def("live_count", &c_api::live_count);
    m.def("double_destroy_count", &c_api::double_destroy_count);

    nb::class_<OwnedHandle>(m, "OwnedHandle")
        .def(nb::init<>())
        .def("get", &OwnedHandle::get)
        .def("owns", &OwnedHandle::owns)
        .def("release", &OwnedHandle::release);

    m.def("create_handle", &c_api::create);
    m.def("move_ctor_steals_and_nulls_source", &move_ctor_steals_and_nulls_source);
    m.def("move_assign_steals_and_nulls_source", &move_assign_steals_and_nulls_source);
    m.def("use_borrowed", &use_borrowed, nb::arg("h"));
    m.def("wrap_borrowed_incorrectly", &wrap_borrowed_incorrectly, nb::arg("h"));
    m.def("wrap_borrowed_correctly", &wrap_borrowed_correctly, nb::arg("h"));
}
