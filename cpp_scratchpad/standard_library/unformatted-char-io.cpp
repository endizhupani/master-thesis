#include <cstdio>
using namespace std;

int main()
{
    const int buffSize = 256;
    static char buff[buffSize];
    fputs("Promt: ", stdout);
    fflush(stdout);
    fgets(buff, buffSize, stdin);
    fputs(buff, stdout);
    return 0;
}