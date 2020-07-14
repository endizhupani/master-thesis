#include <cstdio>

using namespace std;

constexpr int max_string = 1024;
constexpr int repeat = 5;

int main(int argc, char **argv)
{
    const char *fn = "data/testfile.txt";
    const char *str = "This is a literal c-string\n";

    printf("writing file\n");
    FILE *fw = fopen(fn, "w");

    for (size_t i = 0; i < repeat; i++)
    {
        fputs(str, fw);
    }

    fclose(fw);

    printf("done\n");
    printf("reading the file");
    char buf[max_string];

    FILE *fr = fopen(fn, "r");

    while (fgets(buf, max_string, fr))
    {
        fputs(buf, stdout);
    }

    fclose(fr);
    remove(fn);

    printf("done.\n");

    return 0;
}