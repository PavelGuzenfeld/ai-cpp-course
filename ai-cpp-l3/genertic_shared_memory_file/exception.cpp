#include <array>
#include <cstdint>
#include <exception>
#include <fmt/format.h>
#include <span>
#include <string>
extern "C"
{

    std::array<std::uint8_t, 256> storage{};
    void *__cxa_allocate_exception(unsigned long /*p_size*/) noexcept
    {
        static constexpr size_t offset = 128;
        return storage.data() + offset;
    }
    void __cxa_free_exception(void *) noexcept
    {
        // Do nothing here.
    }

    std::array<std::byte, 1024> heap_memory{};
    std::span<std::byte> available_memory = heap_memory;
    void *_sbrk(std::size_t p_amount)
    {
        if (p_amount > available_memory.size())
        {
            return nullptr;
        }
        auto result = available_memory.subspan(0, p_amount);
        available_memory = available_memory.subspan(p_amount);
        return result.data();
    }
    namespace __cxxabiv1
    {
        std::terminate_handler __terminate_handler = +[]()
        {
            while (true)
            {
                continue;
            }
        };
    } // namespace __cxxabiv1
}

[[gnu::noinline]] int start() { throw std::string("Hello world"); }
int main()
{
    volatile int return_code = 0;
    try
    {
        return_code = start();
    }
    catch (std::string const &exception)
    {
        fmt::print("The exception is {}.\n", exception);
        return_code = -1;
    }
    return return_code;
}
