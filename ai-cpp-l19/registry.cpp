#include "registry.hpp"

#include <atomic>

namespace linking_demo
{
namespace
{
std::atomic<int> g_touch_count{0};
}

int register_touch()
{
    return g_touch_count.fetch_add(1, std::memory_order_relaxed) + 1;
}

int current_count()
{
    return g_touch_count.load(std::memory_order_relaxed);
}

} // namespace linking_demo
