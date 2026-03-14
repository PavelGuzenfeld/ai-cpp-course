#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace nb = nanobind;

struct TimingRecord
{
    std::string name;
    int64_t elapsed_ns;
};

class Timer
{
public:
    static Timer &instance()
    {
        static Timer t;
        return t;
    }

    void record(const std::string &name, int64_t elapsed_ns)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back({name, elapsed_ns});
        by_name_[name].push_back(elapsed_ns);
    }

    std::vector<std::string> section_names() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        names.reserve(by_name_.size());
        for (const auto &[name, _] : by_name_)
        {
            names.push_back(name);
        }
        return names;
    }

    std::vector<int64_t> timings_for(const std::string &name) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_name_.find(name);
        if (it == by_name_.end())
        {
            return {};
        }
        return it->second;
    }

    std::vector<int64_t> all_timings_ns() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<int64_t> result;
        result.reserve(records_.size());
        for (const auto &r : records_)
        {
            result.push_back(r.elapsed_ns);
        }
        return result;
    }

    std::vector<std::string> all_names() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> result;
        result.reserve(records_.size());
        for (const auto &r : records_)
        {
            result.push_back(r.name);
        }
        return result;
    }

    size_t count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.size();
    }

    size_t count_for(const std::string &name) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_name_.find(name);
        if (it == by_name_.end())
        {
            return 0;
        }
        return it->second.size();
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        by_name_.clear();
    }

private:
    Timer() = default;
    mutable std::mutex mutex_;
    std::vector<TimingRecord> records_;
    std::unordered_map<std::string, std::vector<int64_t>> by_name_;
};

class ScopedTimer
{
public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name)),
          start_(std::chrono::steady_clock::now())
    {
    }

    ~ScopedTimer()
    {
        stop();
    }

    void stop()
    {
        if (!stopped_)
        {
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
            Timer::instance().record(name_, elapsed);
            stopped_ = true;
        }
    }

    int64_t elapsed_ns() const
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_).count();
    }

    // Context manager support
    ScopedTimer &enter() { return *this; }
    void exit() { stop(); }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};

int64_t measure_steady_clock_ns()
{
    auto start = std::chrono::steady_clock::now();
    // Measure the overhead of a clock read itself
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

NB_MODULE(latency_timer, m)
{
    m.doc() = "High-resolution latency timer using std::chrono::steady_clock";

    nb::class_<ScopedTimer>(m, "ScopedTimer")
        .def(nb::init<std::string>(), nb::arg("name"))
        .def("stop", &ScopedTimer::stop, "Stop the timer and record the measurement")
        .def("elapsed_ns", &ScopedTimer::elapsed_ns, "Get elapsed time without stopping")
        .def("__enter__", &ScopedTimer::enter, nb::rv_policy::reference)
        .def("__exit__",
             [](ScopedTimer &self, nb::handle, nb::handle, nb::handle)
             {
                 self.stop();
             });

    nb::class_<Timer>(m, "Timer")
        .def_static("instance", &Timer::instance, nb::rv_policy::reference)
        .def("record", &Timer::record, nb::arg("name"), nb::arg("elapsed_ns"))
        .def("section_names", &Timer::section_names)
        .def("timings_for", &Timer::timings_for, nb::arg("name"))
        .def("all_timings_ns", &Timer::all_timings_ns)
        .def("all_names", &Timer::all_names)
        .def("count", &Timer::count)
        .def("count_for", &Timer::count_for, nb::arg("name"))
        .def("reset", &Timer::reset);

    m.def("measure_steady_clock_ns", &measure_steady_clock_ns,
          "Measure the overhead of reading steady_clock (in nanoseconds)");
}
