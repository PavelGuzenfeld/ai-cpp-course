#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <chrono>
#include <optional>
#include <string>
#include <tuple>
#include <variant>

namespace nb = nanobind;

// ===========================================================================
// "BEFORE": String-based state machine (mirrors tracker_engine pattern)
// ===========================================================================
class StringStateMachine {
public:
    StringStateMachine() : state_{"idle"}, lost_frames_{0} {}

    void update(bool has_detection, float det_x = 0, float det_y = 0,
                float det_w = 0, float det_h = 0) {
        if (state_ == "idle") {
            if (has_detection) {
                state_ = "tracking";
                target_ = {det_x, det_y, det_w, det_h};
            }
        } else if (state_ == "tracking") {
            if (has_detection) {
                target_ = {det_x, det_y, det_w, det_h};
            } else {
                state_ = "lost";
                lost_frames_ = 1;
                last_known_ = target_;
            }
        } else if (state_ == "lost") {
            if (has_detection) {
                state_ = "tracking";
                target_ = {det_x, det_y, det_w, det_h};
                lost_frames_ = 0;
            } else {
                lost_frames_++;
                if (lost_frames_ > 30) {
                    state_ = "search";
                }
            }
        } else if (state_ == "search") {
            if (has_detection) {
                state_ = "tracking";
                target_ = {det_x, det_y, det_w, det_h};
                lost_frames_ = 0;
            }
            // else: stay in search
        }
    }

    [[nodiscard]] std::string state() const { return state_; }
    [[nodiscard]] int lost_frames() const { return lost_frames_; }
    [[nodiscard]] std::tuple<float, float, float, float> target() const { return target_; }

private:
    std::string state_;
    int lost_frames_;
    std::tuple<float, float, float, float> target_{0, 0, 0, 0};
    std::tuple<float, float, float, float> last_known_{0, 0, 0, 0};
};

// ===========================================================================
// "AFTER": std::variant + std::visit state machine
// ===========================================================================

struct BBox {
    float x, y, w, h;
};

// Each state is its own type, carrying only the data relevant to that state.
struct Idle {};
struct Tracking { BBox target; };
struct Lost { int frames_lost; BBox last_known; };
struct Search { BBox last_known; };

using TrackerState = std::variant<Idle, Tracking, Lost, Search>;

// Helper for overloaded lambdas in std::visit
template <class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

class VariantStateMachine {
public:
    VariantStateMachine() : state_{Idle{}} {}

    void update(bool has_detection, float det_x = 0, float det_y = 0,
                float det_w = 0, float det_h = 0) {
        std::optional<BBox> detection;
        if (has_detection) {
            detection = BBox{det_x, det_y, det_w, det_h};
        }

        state_ = std::visit(overloaded{
            [&](Idle) -> TrackerState {
                if (detection) return Tracking{*detection};
                return Idle{};
            },
            [&](Tracking& t) -> TrackerState {
                if (detection) return Tracking{*detection};
                return Lost{1, t.target};
            },
            [&](Lost& l) -> TrackerState {
                if (detection) return Tracking{*detection};
                if (l.frames_lost > 30) return Search{l.last_known};
                return Lost{l.frames_lost + 1, l.last_known};
            },
            [&](Search& s) -> TrackerState {
                if (detection) return Tracking{*detection};
                return Search{s.last_known};
            }
        }, state_);
    }

    [[nodiscard]] std::string state() const {
        return std::visit(overloaded{
            [](const Idle&) -> std::string { return "idle"; },
            [](const Tracking&) -> std::string { return "tracking"; },
            [](const Lost&) -> std::string { return "lost"; },
            [](const Search&) -> std::string { return "search"; }
        }, state_);
    }

    [[nodiscard]] int lost_frames() const {
        return std::visit(overloaded{
            [](const Lost& l) -> int { return l.frames_lost; },
            [](const auto&) -> int { return 0; }
        }, state_);
    }

    [[nodiscard]] std::tuple<float, float, float, float> target() const {
        return std::visit(overloaded{
            [](const Tracking& t) -> std::tuple<float, float, float, float> {
                return {t.target.x, t.target.y, t.target.w, t.target.h};
            },
            [](const Lost& l) -> std::tuple<float, float, float, float> {
                return {l.last_known.x, l.last_known.y, l.last_known.w, l.last_known.h};
            },
            [](const Search& s) -> std::tuple<float, float, float, float> {
                return {s.last_known.x, s.last_known.y, s.last_known.w, s.last_known.h};
            },
            [](const auto&) -> std::tuple<float, float, float, float> {
                return {0, 0, 0, 0};
            }
        }, state_);
    }

private:
    TrackerState state_;
};

// ===========================================================================
// Benchmark helpers (run N iterations of state transitions)
// ===========================================================================
double benchmark_string_sm(int iterations) {
    StringStateMachine sm;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        // Simulate a tracking scenario:
        // idle -> tracking -> tracking -> lost -> ... -> search -> tracking
        sm.update(true, 100.0f, 200.0f, 50.0f, 50.0f);   // idle -> tracking
        sm.update(true, 105.0f, 205.0f, 50.0f, 50.0f);   // tracking -> tracking
        sm.update(false);                                    // tracking -> lost
        for (int j = 0; j < 31; ++j) {
            sm.update(false);                                // lost -> lost -> ... -> search
        }
        sm.update(true, 110.0f, 210.0f, 50.0f, 50.0f);   // search -> tracking
        sm.update(false);                                    // tracking -> lost
        sm.update(true, 115.0f, 215.0f, 50.0f, 50.0f);   // lost -> tracking
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

double benchmark_variant_sm(int iterations) {
    VariantStateMachine sm;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        sm.update(true, 100.0f, 200.0f, 50.0f, 50.0f);
        sm.update(true, 105.0f, 205.0f, 50.0f, 50.0f);
        sm.update(false);
        for (int j = 0; j < 31; ++j) {
            sm.update(false);
        }
        sm.update(true, 110.0f, 210.0f, 50.0f, 50.0f);
        sm.update(false);
        sm.update(true, 115.0f, 215.0f, 50.0f, 50.0f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// ===========================================================================
// Nanobind bindings
// ===========================================================================
NB_MODULE(state_machine, m) {
    m.doc() = "String-based vs variant-based state machine comparison";

    // --- String state machine ("before") ---
    nb::class_<StringStateMachine>(m, "StringStateMachine")
        .def(nb::init<>())
        .def("update", &StringStateMachine::update,
             nb::arg("has_detection"),
             nb::arg("det_x") = 0.0f, nb::arg("det_y") = 0.0f,
             nb::arg("det_w") = 0.0f, nb::arg("det_h") = 0.0f)
        .def("state", &StringStateMachine::state)
        .def("lost_frames", &StringStateMachine::lost_frames)
        .def("target", &StringStateMachine::target);

    // --- Variant state machine ("after") ---
    nb::class_<VariantStateMachine>(m, "VariantStateMachine")
        .def(nb::init<>())
        .def("update", &VariantStateMachine::update,
             nb::arg("has_detection"),
             nb::arg("det_x") = 0.0f, nb::arg("det_y") = 0.0f,
             nb::arg("det_w") = 0.0f, nb::arg("det_h") = 0.0f)
        .def("state", &VariantStateMachine::state)
        .def("lost_frames", &VariantStateMachine::lost_frames)
        .def("target", &VariantStateMachine::target);

    // --- Benchmarks ---
    m.def("benchmark_string_sm", &benchmark_string_sm, nb::arg("iterations"),
          "Benchmark string-based state machine (returns microseconds)");
    m.def("benchmark_variant_sm", &benchmark_variant_sm, nb::arg("iterations"),
          "Benchmark variant-based state machine (returns microseconds)");
}
