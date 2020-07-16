#include <cstdio>
#include <cstring>

using namespace std;

int main()
{
    const static size_t maxbuf = 128;
    const char *s1 = "String one";
    const char *s2 = "String two";

    char sd1[maxbuf];
    char sd2[maxbuf];
    int i = 0;
    char c = 0;
    char *cp = nullptr;

    strncpy(sd1, s1, maxbuf);
    printf("sd1 is %s\n", sd1);
    strncpy(sd2, s2, maxbuf);
    printf("sd2 is %s\n", sd2);

    strncat(sd1, " - ", maxbuf - strlen(sd1) - 1);
    strncat(sd1, s2, maxbuf - strlen(sd1) - 1);

    printf("sd1 is %s\n", sd1);

    return 0;
}