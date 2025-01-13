#include <nanobind/nanobind.h>
#include <stdexcept> // because we don't trust you to behave

namespace nb = nanobind;

class MyClass
{
public:
    MyClass(int x) : value_{x}
    {
        if (x < 0)
        {
            throw std::invalid_argument("value must be non-negative");
        }
    }

    [[nodiscard]] int get_value() const noexcept
    {
        return value_;
    }

    void set_value(int x)
    {
        if (x < 0)
        {
            throw std::invalid_argument("value must be non-negative");
        }
        value_ = x;
    }

private:
    int value_;
};

NB_MODULE(my_module, m)
{
    nb::class_<MyClass>(m, "MyClass")
        .def(nb::init<int>(), nb::arg("value") = 0) // default value
        .def("get_value", &MyClass::get_value)
        .def("set_value", &MyClass::set_value);
}
