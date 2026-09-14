#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "flat-type/flat.hpp"
#include "shm/shm.hpp"

namespace nb = nanobind;

namespace
{
    struct SensorReading
    {
        double x;
        double y;
        double z;
        std::uint64_t seq;
    };
    static_assert(FlatType<SensorReading>,
                  "SensorReading must satisfy FlatType to live in shared memory");

    class ShmWriter
    {
    public:
        explicit ShmWriter(std::string const &name)
            : segment_(shm::path(name), sizeof(SensorReading))
        {
        }

        void write(double x, double y, double z, std::uint64_t seq)
        {
            auto *slot = static_cast<SensorReading *>(segment_.get());
            *slot = SensorReading{x, y, z, seq};
        }

    private:
        shm::Shm segment_;
    };

    class ShmReader
    {
    public:
        explicit ShmReader(std::string const &name)
            : segment_(shm::path(name), sizeof(SensorReading))
        {
        }

        nb::tuple read() const
        {
            SensorReading snapshot;
            std::memcpy(&snapshot, segment_.get(), sizeof(SensorReading));
            return nb::make_tuple(snapshot.x, snapshot.y, snapshot.z, snapshot.seq);
        }

    private:
        shm::Shm segment_;
    };
} // namespace

NB_MODULE(shm_roundtrip_native, m)
{
    nb::class_<ShmWriter>(m, "ShmWriter")
        .def(nb::init<std::string const &>(), nb::arg("name"))
        .def("write", &ShmWriter::write, nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("seq"));

    nb::class_<ShmReader>(m, "ShmReader")
        .def(nb::init<std::string const &>(), nb::arg("name"))
        .def("read", &ShmReader::read);
}
