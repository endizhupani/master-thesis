#include <cstdio>

using namespace std;

int main()
{
    static const char *fn1 = "file1";
    static const char *fn2 = "file2";
    remove(fn2);
    puts("Done");
    return 0;
}