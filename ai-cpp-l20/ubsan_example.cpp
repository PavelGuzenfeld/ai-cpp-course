// Standalone executable exercising the UB from PRODUCTION_PLAN.md's fix
// commit: casting a non-finite double to int is undefined behaviour, not a
// large-number result. Build with -DENABLE_SANITIZERS=ON (see
// CMakeLists.txt) to have UBSan trap it at runtime instead of silently
// returning garbage.
#include <cstdio>
#include <limits>

namespace
{
    int cast_unsafely(double x)
    {
        return static_cast<int>(x); // UB when x is NaN, +-Inf, or out of int's range
    }
} // namespace

int main()
{
    double const poisoned = std::numeric_limits<double>::quiet_NaN();
    int const result = cast_unsafely(poisoned);
    std::printf("result: %d\n", result);
    return 0;
}
