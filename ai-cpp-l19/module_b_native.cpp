#include <nanobind/nanobind.h>

#include "registry.hpp"

namespace nb = nanobind;

NB_MODULE(module_b_native, m)
{
    m.doc() = "Module B: links linking_registry, same touch-count API as module_a_native";

    m.def("register_touch", &linking_demo::register_touch,
          nb::call_guard<nb::gil_scoped_release>(),
          "Increment the touch counter, return the new count");
    m.def("current_count", &linking_demo::current_count,
          "Read the touch counter without incrementing it");
}
