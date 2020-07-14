#include <cstdio>
#include <string>
using namespace std;

template <typename T>
T maxOf(T a, T b)
{
    return (a > b ? a : b);
}

int main(int argc, char **argv)
{
    string m = maxOf<string>("nine", "seven");
    printf("max is %s\n:", m.c_str());
    return 0;
}