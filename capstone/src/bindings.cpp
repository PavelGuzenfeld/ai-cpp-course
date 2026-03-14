/**
 * pybind11 bindings for fast_tracker_utils.
 *
 * Students: add your component bindings here as you implement them.
 * Each component should have its own .cpp file under src/, and the
 * bindings here expose them to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, m)
{
    m.doc() = "fast_tracker_utils native extension module";

    // Example: uncomment and adapt as you implement each component
    //
    // py::class_<FastKalmanFilter>(m, "FastKalmanFilter")
    //     .def(py::init<int>(), py::arg("state_dim"))
    //     .def("predict", &FastKalmanFilter::predict)
    //     .def("update", &FastKalmanFilter::update);
    //
    // py::class_<FastPreprocessor>(m, "FastPreprocessor")
    //     .def(py::init<>())
    //     .def("process", &FastPreprocessor::process);
    //
    // py::class_<FastHistoryBuffer>(m, "FastHistoryBuffer")
    //     .def(py::init<int, int>(), py::arg("capacity"), py::arg("dim"))
    //     .def("push", &FastHistoryBuffer::push)
    //     .def("latest", &FastHistoryBuffer::latest);
    //
    // py::class_<FastStateMachine>(m, "FastStateMachine")
    //     .def(py::init<>())
    //     .def("process_event", &FastStateMachine::process_event)
    //     .def_property_readonly("state", &FastStateMachine::state);
}
