#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/sanitized_key.hpp"
#include "safe-shm/seqlock.hpp"
#include "safe-shm/shm_lifecycle.hpp"
#include "safe-shm/stamped.hpp"
#include "safe-shm/time_series.hpp"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include <fmt/core.h>

namespace nb = nanobind;
using namespace nb::literals;

// Concrete instantiations
using StampedF64 = safe_shm::Stamped<double>;

NB_MODULE(safe_shm_py, m)
{
    m.doc() = "safe-shm Python bindings: lock-free shared memory primitives";

    // ── Stamped<double> ──────────────────────────────────────────────────
    nb::class_<StampedF64>(m, "StampedF64")
        .def(nb::init<>())
        .def_rw("timestamp_ns", &StampedF64::timestamp_ns)
        .def_rw("sequence", &StampedF64::sequence)
        .def_rw("data", &StampedF64::data)
        .def("__repr__", [](StampedF64 const &s) -> std::string
             { return fmt::format("StampedF64(ts={}, seq={}, data={})",
                                  s.timestamp_ns, s.sequence, s.data); });

    m.def("stamp_f64", [](double data, uint64_t seq) -> StampedF64
          { return safe_shm::stamp(data, seq); },
          "data"_a, "seq"_a,
          "Create a Stamped<double> with current monotonic timestamp");

    m.def("monotonic_now_ns", &safe_shm::monotonic_now_ns,
          "Get current CLOCK_MONOTONIC time in nanoseconds");

    // ── SeqlockWriter<double> ────────────────────────────────────────────
    nb::class_<safe_shm::SeqlockWriter<double>>(m, "SeqlockWriterF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        .def("store", &safe_shm::SeqlockWriter<double>::store, "data"_a);

    // ── SeqlockReader<double> ────────────────────────────────────────────
    nb::class_<safe_shm::SeqlockReader<double>>(m, "SeqlockReaderF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        .def("load", &safe_shm::SeqlockReader<double>::load)
        .def("sequence", &safe_shm::SeqlockReader<double>::sequence);

    // ── SeqlockWriter<Stamped<double>> ───────────────────────────────────
    nb::class_<safe_shm::SeqlockWriter<StampedF64>>(m, "SeqlockWriterStampedF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        .def("store", &safe_shm::SeqlockWriter<StampedF64>::store, "data"_a);

    // ── SeqlockReader<Stamped<double>> ───────────────────────────────────
    nb::class_<safe_shm::SeqlockReader<StampedF64>>(m, "SeqlockReaderStampedF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        .def("load", &safe_shm::SeqlockReader<StampedF64>::load)
        .def("sequence", &safe_shm::SeqlockReader<StampedF64>::sequence);

    // ── CyclicBufferWriter<Stamped<double>, 64> ─────────────────────────
    using CBWriter = safe_shm::CyclicBufferWriter<StampedF64, 64>;
    nb::class_<CBWriter>(m, "CyclicBufferWriterStampedF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        .def("insert", &CBWriter::insert, "data"_a)
        .def("total_writes", &CBWriter::total_writes);

    // ── CyclicBufferReader<Stamped<double>, 64> ─────────────────────────
    using CBReader = safe_shm::CyclicBufferReader<StampedF64, 64>;
    nb::class_<CBReader>(m, "CyclicBufferReaderStampedF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        .def("get_latest", &CBReader::get_latest)
        .def("get", &CBReader::get, "reverse_index"_a)
        .def("try_get", &CBReader::try_get, "reverse_index"_a, "max_retries"_a = 4u)
        .def("available", &CBReader::available)
        .def("total_writes", &CBReader::total_writes)
        .def_prop_ro_static("capacity", [](nb::handle) { return CBReader::capacity(); });

    // ── TimeSeries<Stamped<double>, 64> ──────────────────────────────────
    using TS = safe_shm::TimeSeries<StampedF64, 64>;
    using InterpPairStamped = safe_shm::InterpPair<StampedF64>;

    nb::class_<InterpPairStamped>(m, "InterpPairStampedF64")
        .def_ro("before", &InterpPairStamped::before)
        .def_ro("after", &InterpPairStamped::after)
        .def_ro("alpha", &InterpPairStamped::alpha);

    nb::class_<TS>(m, "TimeSeriesStampedF64")
        .def(nb::init<std::string>(), "shm_name"_a)
        // Use lambdas to disambiguate from SanitizedKey<Tag> template overloads
        .def("find_closest",
             [](TS const &self, uint64_t target, std::optional<uint64_t> max_dist) {
                 return self.find_closest(target, max_dist);
             },
             "target_timestamp_ns"_a, "max_distance_ns"_a = nb::none())
        .def("find_interpolation_pair",
             [](TS const &self, uint64_t target) {
                 return self.find_interpolation_pair(target);
             },
             "target_timestamp_ns"_a)
        .def("get_latest_if_fresh",
             [](TS const &self, uint64_t min_key) {
                 return self.get_latest_if_fresh(min_key);
             },
             "min_timestamp_ns"_a)
        .def("get_latest", &TS::get_latest)
        .def("available", &TS::available)
        .def("total_writes", &TS::total_writes)
        .def_prop_ro_static("capacity", [](nb::handle) { return TS::capacity(); });

    // ── Sanitizer / Validator utilities (Python-side) ─────────────────────
    m.def(
        "validate_monotonic_ns",
        [](uint64_t v) -> uint64_t {
            if (!safe_shm::MonotonicNsValidator{}(v))
                throw std::invalid_argument(
                    "Value " + std::to_string(v) +
                    " is not a plausible CLOCK_MONOTONIC nanosecond timestamp "
                    "(expected >= 1s, got " + std::to_string(v) + " ns)");
            return v;
        },
        "value"_a,
        "Validate that a value is a plausible CLOCK_MONOTONIC nanosecond timestamp. "
        "Raises ValueError if not.");

    m.def(
        "validate_range",
        [](uint64_t v, uint64_t min_val, uint64_t max_val) -> uint64_t {
            if (!safe_shm::RangeValidator{min_val, max_val}(v))
                throw std::invalid_argument(
                    "Value " + std::to_string(v) +
                    " is outside range [" + std::to_string(min_val) +
                    ", " + std::to_string(max_val) + "]");
            return v;
        },
        "value"_a, "min_val"_a, "max_val"_a,
        "Validate that a value is within [min_val, max_val]. Raises ValueError if not.");

    m.def(
        "validate_proximity",
        [](uint64_t v, uint64_t reference, uint64_t tolerance) -> uint64_t {
            if (!safe_shm::ProximityValidator{reference, tolerance}(v))
                throw std::invalid_argument(
                    "Value " + std::to_string(v) +
                    " differs from reference " + std::to_string(reference) +
                    " by more than tolerance " + std::to_string(tolerance));
            return v;
        },
        "value"_a, "reference"_a, "tolerance"_a,
        "Validate that a value is within tolerance of a reference. "
        "Catches unit mismatches (e.g., ns vs ms). Raises ValueError if not.");

    // ── SHM Lifecycle utilities ──────────────────────────────────────────
    m.def("shm_exists", &safe_shm::shm_exists, "name"_a);
    m.def("shm_remove", &safe_shm::shm_remove, "name"_a);
    m.def("shm_list", &safe_shm::shm_list, "prefix"_a = "");

    nb::class_<safe_shm::ShmHeader>(m, "ShmHeader")
        .def_ro("magic", &safe_shm::ShmHeader::magic)
        .def_ro("version", &safe_shm::ShmHeader::version)
        .def_ro("heartbeat_ns", &safe_shm::ShmHeader::heartbeat_ns)
        .def_ro("writer_pid", &safe_shm::ShmHeader::writer_pid);

    m.def("is_writer_alive", &safe_shm::is_writer_alive,
          "header"_a, "max_stale_ns"_a = uint64_t{0});
}
