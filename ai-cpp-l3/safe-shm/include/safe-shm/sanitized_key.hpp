#pragma once
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace safe_shm
{
    /// Strong type for a temporal key that has been validated through an
    /// external sanitizer function. Cannot be constructed directly —
    /// must go through sanitize().
    ///
    /// Tag type makes different sanitization policies type-incompatible:
    ///   SanitizedKey<MonotonicNsTag> cannot be passed where
    ///   SanitizedKey<RangeCheckedTag> is expected.
    template <typename Tag = void>
    class SanitizedKey
    {
    public:
        using value_type = uint64_t;

        constexpr uint64_t value() const noexcept { return value_; }

        // No implicit conversion to uint64_t — that would bypass the point.
        // Use .value() explicitly when you need the raw number.

    private:
        uint64_t value_;
        explicit constexpr SanitizedKey(uint64_t v) noexcept : value_(v) {}

        template <typename T, typename F>
        friend constexpr auto sanitize(uint64_t raw, F &&fn)
            -> std::optional<SanitizedKey<T>>;
    };

    /// The only way to create a SanitizedKey: pass the raw value through
    /// a validator function. Returns nullopt if validation fails.
    ///
    /// Usage:
    ///   auto key = sanitize<MyTag>(raw_ns, MonotonicNsValidator{});
    ///   auto key = sanitize<MyTag>(raw_ns, [](uint64_t v) { return v > 0; });
    template <typename Tag = void, typename F>
    constexpr auto sanitize(uint64_t raw, F &&fn) -> std::optional<SanitizedKey<Tag>>
    {
        if (std::forward<F>(fn)(raw))
            return SanitizedKey<Tag>(raw);
        return std::nullopt;
    }

    // ── Built-in validator functions ──────────────────────────────────────

    /// Rejects values outside [min_val, max_val].
    struct RangeValidator
    {
        uint64_t min_val;
        uint64_t max_val;

        constexpr bool operator()(uint64_t v) const noexcept
        {
            return v >= min_val && v <= max_val;
        }
    };

    /// Rejects values that differ from a reference by more than tolerance.
    /// Catches unit mismatches (e.g., ns vs ms: reference=5e9, value=5000).
    struct ProximityValidator
    {
        uint64_t reference;
        uint64_t tolerance;

        constexpr bool operator()(uint64_t v) const noexcept
        {
            return (v > reference ? v - reference : reference - v) <= tolerance;
        }
    };

    /// Rejects values that are not plausible CLOCK_MONOTONIC nanoseconds.
    /// A CLOCK_MONOTONIC value is typically between ~1 second (just booted)
    /// and ~100 years of uptime.
    struct MonotonicNsValidator
    {
        static constexpr uint64_t MIN_NS = 1'000'000'000ULL;                        // 1 second
        static constexpr uint64_t MAX_NS = 100ULL * 365 * 86400 * 1'000'000'000ULL; // ~100 years

        constexpr bool operator()(uint64_t v) const noexcept
        {
            return v >= MIN_NS && v <= MAX_NS;
        }
    };

    /// Combines multiple validators with AND logic.
    template <typename... Validators>
    struct AllOf
    {
        std::tuple<Validators...> validators;

        constexpr AllOf(Validators... vs) noexcept : validators(std::move(vs)...) {}

        constexpr bool operator()(uint64_t v) const noexcept
        {
            return std::apply(
                [v](auto const &...vs) { return (vs(v) && ...); },
                validators);
        }
    };

    // Deduction guide
    template <typename... Vs>
    AllOf(Vs...) -> AllOf<Vs...>;

    // ── Tag types ────────────────────────────────────────────────────────
    // Different tags make different policies type-incompatible at compile time.

    struct MonotonicNsTag
    {
    };
    struct RangeCheckedTag
    {
    };
    struct ProximityCheckedTag
    {
    };

} // namespace safe_shm
