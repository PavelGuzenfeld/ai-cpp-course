#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "record_parser.hpp"

namespace nb = nanobind;

struct ParsedRecord
{
    std::uint8_t version;
    std::vector<std::uint8_t> payload;
};

// Same code path the TSan lane exercises from multiple Python threads, so it
// must actually run concurrently to mean anything under the sanitizer. The
// GIL is released only around record_parser::parse() itself, which never
// touches a Python object -- reading nb::bytes's buffer, and building the
// std::optional<ParsedRecord> return value nanobind converts to Python,
// happen with the GIL held.
std::optional<ParsedRecord> parse_record(nb::bytes data)
{
    const auto *ptr = reinterpret_cast<const std::uint8_t *>(data.c_str());
    std::size_t size = data.size();

    std::optional<record_parser::Record> result;
    {
        nb::gil_scoped_release release;
        result = record_parser::parse(ptr, size);
    }

    if (!result)
    {
        return std::nullopt;
    }
    return ParsedRecord{result->version, std::move(result->payload)};
}

NB_MODULE(record_parser_native, m)
{
    m.doc() = "Bounds-checked binary record parser, fuzzed in fuzz/fuzz_parser.cpp";

    nb::class_<ParsedRecord>(m, "ParsedRecord")
        .def_ro("version", &ParsedRecord::version)
        .def_ro("payload", &ParsedRecord::payload);

    m.def("parse_record", &parse_record, nb::arg("data").noconvert(),
          "Parse a record; returns None for anything malformed instead of raising");
}
