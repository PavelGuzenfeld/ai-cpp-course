#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/stamped.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

// Global counter for unique SHM names across fuzzer invocations.
static std::atomic<uint64_t> g_counter{0};

static std::string unique_name(char const *prefix)
{
    return std::string(prefix) + "_" + std::to_string(getpid()) + "_" +
           std::to_string(g_counter.fetch_add(1, std::memory_order_relaxed));
}

// Consume a value of type U from the fuzz input, advancing the pointer.
// Returns false if not enough bytes remain.
template <typename U>
static bool consume(const uint8_t *&ptr, size_t &remaining, U &out)
{
    if (remaining < sizeof(U))
        return false;
    std::memcpy(&out, ptr, sizeof(U));
    ptr += sizeof(U);
    remaining -= sizeof(U);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    using Elem = safe_shm::Stamped<double>;
    constexpr std::size_t Cap = 16;

    // --- Zero-length input: exercise empty-buffer paths ---
    if (size == 0)
    {
        auto name = unique_name("fuzz_cb_empty");
        safe_shm::CyclicBuffer<Elem, Cap> buf(name);

        // Reading from empty buffer should not crash.
        (void)buf.get_latest();
        (void)buf.available();
        (void)buf.total_writes();
        (void)buf.try_get(0);
        (void)buf.try_get(999);
        return 0;
    }

    // --- Same-process CyclicBuffer fuzz ---
    {
        auto name = unique_name("fuzz_cb_same");
        safe_shm::CyclicBuffer<Elem, Cap> buf(name);

        const uint8_t *ptr = data;
        size_t remaining = size;

        while (remaining > 0)
        {
            uint8_t op{};
            if (!consume(ptr, remaining, op))
                break;

            switch (op % 6)
            {
            case 0: // Insert with fuzzed timestamp and value
            {
                uint64_t ts{};
                double val{};
                uint64_t seq{};
                if (!consume(ptr, remaining, ts))
                    break;
                if (!consume(ptr, remaining, val))
                    val = 0.0;
                if (!consume(ptr, remaining, seq))
                    seq = 0;
                Elem e{ts, seq, val};
                buf.insert(e);
                break;
            }
            case 1: // get_latest
            {
                (void)buf.get_latest();
                break;
            }
            case 2: // get by reverse index (fuzzed)
            {
                uint8_t idx{};
                if (!consume(ptr, remaining, idx))
                    break;
                if (buf.available() > 0)
                {
                    auto ri = static_cast<std::size_t>(idx) % buf.available();
                    (void)buf.get(ri);
                }
                break;
            }
            case 3: // try_get with arbitrary index (may be out of range)
            {
                uint16_t idx{};
                if (!consume(ptr, remaining, idx))
                    break;
                (void)buf.try_get(static_cast<std::size_t>(idx));
                break;
            }
            case 4: // available + total_writes
            {
                auto avail = buf.available();
                auto tw = buf.total_writes();
                // Invariant: available <= capacity
                if (avail > Cap)
                    __builtin_trap();
                // Invariant: available <= total_writes
                if (avail > tw)
                    __builtin_trap();
                break;
            }
            case 5: // Insert with extreme timestamps
            {
                uint8_t which{};
                if (!consume(ptr, remaining, which))
                    break;
                uint64_t ts{};
                switch (which % 4)
                {
                case 0:
                    ts = 0;
                    break;
                case 1:
                    ts = UINT64_MAX;
                    break;
                case 2:
                    ts = UINT64_MAX / 2;
                    break;
                case 3:
                    ts = 1;
                    break;
                }
                Elem e{ts, static_cast<uint64_t>(which), 3.14};
                buf.insert(e);
                break;
            }
            }
        }

        // Final consistency checks after all operations.
        auto avail = buf.available();
        if (avail > Cap)
            __builtin_trap();
        if (avail > 0)
        {
            (void)buf.get_latest();
            (void)buf.get(0);
            if (avail > 1)
                (void)buf.get(avail - 1);
        }
    }

    // --- Cross-process Writer + Reader fuzz ---
    {
        auto name = unique_name("fuzz_cb_wr");
        safe_shm::CyclicBufferWriter<Elem, Cap> writer(name);
        safe_shm::CyclicBufferReader<Elem, Cap> reader(name);

        const uint8_t *ptr = data;
        size_t remaining = size;

        uint64_t insert_count = 0;

        while (remaining > 0)
        {
            uint8_t op{};
            if (!consume(ptr, remaining, op))
                break;

            switch (op % 4)
            {
            case 0: // Insert
            {
                uint64_t ts{};
                double val{};
                if (!consume(ptr, remaining, ts))
                    break;
                if (!consume(ptr, remaining, val))
                    val = 0.0;
                Elem e{ts, insert_count, val};
                writer.insert(e);
                ++insert_count;
                break;
            }
            case 1: // Reader get_latest
            {
                (void)reader.get_latest();
                break;
            }
            case 2: // Reader try_get
            {
                uint8_t idx{};
                if (!consume(ptr, remaining, idx))
                    break;
                (void)reader.try_get(static_cast<std::size_t>(idx));
                break;
            }
            case 3: // Reader available + total_writes consistency
            {
                auto avail = reader.available();
                auto tw = reader.total_writes();
                if (avail > Cap)
                    __builtin_trap();
                if (avail > tw)
                    __builtin_trap();
                break;
            }
            }
        }
    }

    // --- Massive overflow: insert far more than capacity ---
    {
        auto name = unique_name("fuzz_cb_overflow");
        safe_shm::CyclicBuffer<Elem, Cap> buf(name);

        // Use all fuzz bytes as a stream of inserts.
        uint64_t seq = 0;
        for (size_t i = 0; i + sizeof(double) <= size; i += sizeof(double))
        {
            double val{};
            std::memcpy(&val, data + i, sizeof(double));
            Elem e{seq, seq, val};
            buf.insert(e);
            ++seq;
        }

        // After overflow, available must be min(seq, Cap).
        auto avail = buf.available();
        auto expected = (seq < Cap) ? seq : Cap;
        if (avail != expected)
            __builtin_trap();

        // Read every available slot.
        for (std::size_t i = 0; i < avail; ++i)
        {
            auto elem = buf.try_get(i);
            if (!elem)
                __builtin_trap(); // single-writer, should always succeed
        }

        // Out-of-range try_get must return nullopt.
        if (buf.try_get(avail).has_value())
            __builtin_trap();
    }

    return 0;
}
