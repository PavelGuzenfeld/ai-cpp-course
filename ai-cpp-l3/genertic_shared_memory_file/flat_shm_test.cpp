#include "flat_shared_memory.hpp"
#include <assert.h>

// constexpr auto const SHARED_MEM_IMAGE_4k_SIZE = 3 * 3840 * 2160;

int main()
{
    using namespace flat_shm;                
    
    {
        fmt::print("Test SharedMemory with int\n");
        auto shared_memory = SharedMemory<int>::create("int_file_name");
        shared_memory->write(42);
        auto read_int = shared_memory->read();
        assert(read_int == 42 && "Failed to read int");
    }

    {
        fmt::print("Test SharedMemory with double\n");
        auto shared_memory = SharedMemory<double>::create("double_file_name");
        shared_memory->write(42.42);
        auto read_double = shared_memory->read();
        assert(read_double == 42.42 && "Failed to read double");
    }

    {
        fmt::print("Test SharedMemory with char\n");
        auto shared_memory = SharedMemory<char>::create("char_file_name");
        shared_memory->write('c');
        auto read_char = shared_memory->read();
        assert(read_char == 'c' && "Failed to read char");
    }

    {
        fmt::print("Test SharedMemory with struct\n");
        struct FlatStruct
        {
            int a;
            double b;
            char buffer[50]; // Correct declaration of a fixed-size array
        };
        auto shared_memory = SharedMemory<FlatStruct>::create("struct_file_name");
        shared_memory->write({42, 42.42, "Hello, shared memory!"});
        auto read_struct = shared_memory->read();
        assert(read_struct.a == 42 && "Failed to read struct.a");
        assert(read_struct.b == 42.42 && "Failed to read struct.b");
        assert(std::string(read_struct.buffer) == "Hello, shared memory!" && "Failed to read struct.buffer");
        assert(shared_memory->size() == sizeof(FlatStruct) && "Failed to get size of struct");
        assert(shared_memory->path() == "/dev/shm/struct_file_name" && "Failed to get path of struct");

    }

    {
        fmt::print("Test SharedMemory with nested struct\n");

        struct FlatStruct
        {
            int a;
            double b;
            char buffer[50]; // Correct declaration of a fixed-size array
        };
        struct NestedFlatStruct
        {
            FlatStruct inner;
            int c;
        };

        auto shared_memory = SharedMemory<NestedFlatStruct>::create("nested_struct_file_name");
        shared_memory->write({{42, 42.42, "Hello, shared memory!"}, 42});
        auto read_struct = shared_memory->read();
        assert(read_struct.inner.a == 42 && "Failed to read nested struct.inner.a");
        assert(read_struct.inner.b == 42.42 && "Failed to read nested struct.inner.b");
        assert(std::string(read_struct.inner.buffer) == "Hello, shared memory!" && "Failed to read nested struct.inner.buffer");
        assert(read_struct.c == 42 && "Failed to read nested struct.c");
        assert(shared_memory->size() == sizeof(NestedFlatStruct) && "Failed to get size of nested struct");
        assert(shared_memory->path() == "/dev/shm/nested_struct_file_name" && "Failed to get path of nested struct");

    }

    {
        fmt::print("Test SharedMemory with array\n");
        auto shared_memory = SharedMemory<int[10]>::create("array_file_name");
        int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        shared_memory->write(data);
        auto read_data = shared_memory->read();
        for (int i = 0; i < 10; i++)
        {
            assert(read_data[i] == i && "Failed to read array");
        }
        assert(shared_memory->size() == sizeof(int[10]) && "Failed to get size of array");
        assert(shared_memory->path() == "/dev/shm/array_file_name" && "Failed to get path of array");
    }


    {
        fmt::print("Move constructor test\n");
        auto shared_memory = SharedMemory<int>::create("move_constructor_file_name");
        shared_memory->write(42);
        auto shared_memory2 = std::move(shared_memory);
        auto read_int = shared_memory2->read();
        assert(read_int == 42 && "Failed to read int after move constructor");
    }

    {
        fmt::print("Move assignment test\n");
        auto shared_memory = SharedMemory<int>::create("move_assignment_file_name");
        shared_memory->write(42);
        auto shared_memory2 = SharedMemory<int>::create("move_assignment_file_name2");
        shared_memory2 = std::move(shared_memory);
        auto read_int = shared_memory2->read();
        assert(read_int == 42 && "Failed to read int after move assignment");
    }
    


    fmt::print("All tests passed\n");

    return 0;
}
