// asan_example.cpp — Intentionally buggy code for ASAN demonstration
//
// This is a STANDALONE program, not a Python module.
// Build with sanitizers enabled:
//
//   cd build
//   cmake .. -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
//   make asan_example -j$(nproc)
//   ./asan_example          # Runs buggy functions (ASAN will report errors)
//   ./asan_example fixed    # Runs only the fixed versions (clean output)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <string>

// ============================================================================
// Bug 1: Buffer Overflow — writes past the end of an array
// ============================================================================

void buffer_overflow_buggy() {
    printf("\n=== buffer_overflow_buggy ===\n");
    int* array = new int[10];

    // Bug: writing to index 10, but valid indices are 0-9
    for (int i = 0; i <= 10; ++i) {
        array[i] = i * 100;  // ASAN: heap-buffer-overflow at i=10
    }

    printf("array[9] = %d\n", array[9]);
    delete[] array;
}

void buffer_overflow_fixed() {
    printf("\n=== buffer_overflow_fixed ===\n");
    // Fix 1: Use std::vector with range-for (cannot go out of bounds)
    std::vector<int> array(10);
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
        array[i] = i * 100;  // Correct: i goes from 0 to 9
    }
    printf("array[9] = %d\n", array[9]);

    // Fix 2: Use .at() for bounds-checked access
    try {
        int val = array.at(10);  // Throws std::out_of_range
        (void)val;
    } catch (const std::out_of_range& e) {
        printf("Caught out_of_range: %s\n", e.what());
    }
}

// ============================================================================
// Bug 2: Use-After-Free — accesses memory after it has been freed
// ============================================================================

void use_after_free_buggy() {
    printf("\n=== use_after_free_buggy ===\n");
    int* data = new int[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = i + 1;
    }

    printf("Before free: data[0] = %d\n", data[0]);

    delete[] data;

    // Bug: accessing memory after it has been freed
    printf("After free: data[0] = %d\n", data[0]);  // ASAN: use-after-free
}

void use_after_free_fixed() {
    printf("\n=== use_after_free_fixed ===\n");
    // Fix: Use unique_ptr — the pointer is automatically nulled after move
    auto data = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i) {
        data[i] = i + 1;
    }
    printf("data[0] = %d\n", data[0]);

    // When data goes out of scope, memory is freed automatically.
    // You cannot accidentally access it because unique_ptr manages lifetime.

    // To demonstrate "moving away" the ownership:
    auto other = std::move(data);
    // data is now nullptr — accessing it would be a clear bug
    if (!data) {
        printf("data is nullptr after move (safe!)\n");
    }
    printf("other[0] = %d\n", other[0]);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    bool run_fixed_only = false;

    if (argc > 1 && std::string(argv[1]) == "fixed") {
        run_fixed_only = true;
    }

    if (run_fixed_only) {
        printf("Running FIXED versions only (should be clean):\n");
        buffer_overflow_fixed();
        use_after_free_fixed();
        printf("\nAll fixed versions completed without errors.\n");
        return 0;
    }

    printf("Running BUGGY versions (ASAN will report errors):\n");
    printf("Note: ASAN may abort after the first error.\n");
    printf("Run with './asan_example fixed' for clean output.\n");

    // ASAN will likely abort on the first buggy function.
    // Comment out one to test the other.
    buffer_overflow_buggy();
    use_after_free_buggy();

    return 0;
}
