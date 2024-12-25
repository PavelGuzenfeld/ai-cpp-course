#include "flat_type.hpp"
#include <string>

// Test structs
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

int main()
{
    static_assert(FlatType<int>);
    static_assert(FlatType<FlatStruct>);
    static_assert(FlatType<NestedFlatStruct>);
    static_assert(!FlatType<std::string>); // Fails: std::string is not trivially copyable
    return 0;
}