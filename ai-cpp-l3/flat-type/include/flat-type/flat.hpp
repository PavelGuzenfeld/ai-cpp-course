#pragma once
#include <type_traits>

template <typename T>
concept FlatType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;
