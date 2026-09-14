// Standalone executable, built twice (see CMakeLists.txt): once with
// assertions active, once with -DNDEBUG. Demonstrates that `assert` is not
// a guard -- it is a debug-only check that a Release build silently drops.
#include <cassert>
#include <cstdlib>
#include <iostream>

namespace
{
    int validate_dimension(int dim)
    {
        assert(dim > 0 && "dimension must be positive");
        return dim;
    }
} // namespace

int main(int argc, char **argv)
{
    int const dim = argc > 1 ? std::atoi(argv[1]) : -1;
    int const validated = validate_dimension(dim);
    std::cout << "validated dimension: " << validated << "\n";
    return 0;
}
