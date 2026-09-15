#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace record_parser
{

// Wire format, all little-endian, no padding:
//   magic    : 4 bytes, must equal kMagic
//   version  : 1 byte
//   length   : 2 bytes, number of payload bytes that follow
//   checksum : 1 byte, sum of payload bytes mod 256
//   payload  : `length` bytes
constexpr std::uint8_t kMagic[4] = {'R', 'C', '1', 0};
constexpr std::size_t kHeaderSize = 8;

struct Record
{
    std::uint8_t version{};
    std::vector<std::uint8_t> payload;
};

// Parses `size` bytes starting at `data`. Returns std::nullopt for anything
// that does not fit the wire format instead of reading past the buffer —
// this function is the direct target of fuzz/fuzz_parser.cpp.
inline std::optional<Record> parse(const std::uint8_t *data, std::size_t size)
{
    if (data == nullptr || size < kHeaderSize)
    {
        return std::nullopt;
    }
    if (std::memcmp(data, kMagic, sizeof(kMagic)) != 0)
    {
        return std::nullopt;
    }

    std::uint8_t version = data[4];
    std::uint16_t length = static_cast<std::uint16_t>(data[5]) |
                            (static_cast<std::uint16_t>(data[6]) << 8);
    std::uint8_t claimed_checksum = data[7];

    // The whole point of this check: `length` is attacker-controlled and
    // must never be trusted past what actually remains in the buffer.
    if (static_cast<std::size_t>(length) > size - kHeaderSize)
    {
        return std::nullopt;
    }

    const std::uint8_t *payload_begin = data + kHeaderSize;
    std::uint8_t computed_checksum = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        computed_checksum = static_cast<std::uint8_t>(computed_checksum + payload_begin[i]);
    }
    if (computed_checksum != claimed_checksum)
    {
        return std::nullopt;
    }

    return Record{version, std::vector<std::uint8_t>(payload_begin, payload_begin + length)};
}

} // namespace record_parser
