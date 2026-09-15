/**
 * state_machine_native: std::variant + std::visit dispatch instead of the
 * baseline's if/elif string comparisons on every event.
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <variant>

namespace nb = nanobind;

namespace
{
struct Idle
{
};
struct Tracking
{
};
struct Lost
{
};
struct Expired
{
};
using State = std::variant<Idle, Tracking, Lost, Expired>;

template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};

// Cached Python str objects for the 4 possible state names: on_event() is
// called once per frame, and constructing+converting a fresh std::string
// every call was costing as much as the entire rest of the dispatch.
const nb::str &idle_str()
{
    static const nb::str s("idle");
    return s;
}
const nb::str &tracking_str()
{
    static const nb::str s("tracking");
    return s;
}
const nb::str &lost_str()
{
    static const nb::str s("lost");
    return s;
}
const nb::str &expired_str()
{
    static const nb::str s("expired");
    return s;
}
} // namespace

class StateMachine
{
public:
    StateMachine() : state_{Idle{}} {}

    nb::str on_event(const std::string &event)
    {
        ++frames_in_state_;

        state_ = std::visit(
            overloaded{
                [&](Idle) -> State
                {
                    if (event == "detect")
                    {
                        frames_in_state_ = 0;
                        return Tracking{};
                    }
                    return Idle{};
                },
                [&](Tracking) -> State
                {
                    if (event == "detect")
                    {
                        frames_in_state_ = 0;
                        return Tracking{};
                    }
                    if (event == "miss")
                    {
                        frames_in_state_ = 0;
                        return Lost{};
                    }
                    return Tracking{};
                },
                [&](Lost) -> State
                {
                    if (event == "detect")
                    {
                        frames_in_state_ = 0;
                        return Tracking{};
                    }
                    if (event == "miss" && frames_in_state_ > 10)
                    {
                        frames_in_state_ = 0;
                        return Expired{};
                    }
                    return Lost{};
                },
                [&](Expired) -> State
                {
                    if (event == "reset")
                    {
                        frames_in_state_ = 0;
                        return Idle{};
                    }
                    return Expired{};
                },
            },
            state_);

        return state();
    }

    [[nodiscard]] nb::str state() const
    {
        return std::visit(
            overloaded{
                [](const Idle &) -> nb::str { return idle_str(); },
                [](const Tracking &) -> nb::str { return tracking_str(); },
                [](const Lost &) -> nb::str { return lost_str(); },
                [](const Expired &) -> nb::str { return expired_str(); },
            },
            state_);
    }

private:
    State state_;
    int frames_in_state_{0};
};

NB_MODULE(state_machine_native, m)
{
    m.doc() = "Variant-based idle/tracking/lost/expired dispatch (capstone component 4)";

    nb::class_<StateMachine>(m, "StateMachine")
        .def(nb::init<>())
        .def("on_event", &StateMachine::on_event, nb::arg("event"))
        .def("state", &StateMachine::state);
}
