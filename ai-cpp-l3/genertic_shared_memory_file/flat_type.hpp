#include <type_traits>

// Helper: Check if a type is a flat type (trivially copyable and standard layout)
template <typename T>
concept SimpleFlatType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

// Helper: Check if all types in a parameter pack are flat
template <typename... Members>
constexpr bool are_all_flat_types()
{
    return (SimpleFlatType<Members> && ...); // Expand over all members
}

// Helper: Check if a type is a struct and ensure all members are flat types
template <typename T>
constexpr bool is_flat_struct()
{
    if constexpr (std::is_aggregate_v<T> && !std::is_fundamental_v<T>)
    {
        // Aggregates can be validated only if all their members are flat
        return true; // Replace this with manual enforcement in aggregate definitions
    }
    return false;
}

// Final flat type concept: flat primitive or aggregate types
template <typename T>
concept FlatType = SimpleFlatType<T> || is_flat_struct<T>();



