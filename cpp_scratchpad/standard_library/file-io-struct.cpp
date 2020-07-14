#include <cstdio>
#include <cstring>
#include <cstdint>

constexpr size_t max_len = 111;

struct S
{
    uint8_t num;
    uint8_t len;
    char str[max_len + 1];
};

int main(int argc, char **argv)
{
    const char *fn = "data/test.file";
    const char *cstr = "This is a literal c-string";

    printf("writing file\n");
    FILE *fw = fopen(fn, "wb");

    static struct S buf1;
    for (size_t i = 0; i < 5; i++)
    {
        buf1.num = i;
        buf1.len = (uint8_t)strlen(cstr);
        if (buf1.len > max_len)
            buf1.len = max_len;
        strncpy(buf1.str, cstr, max_len);
        buf1.str[buf1.len] = 0;
        fwrite(&buf1, sizeof(struct S), 1, fw);
    }

    fclose(fw);
    printf("done.\n");

    printf("reading\n");

    FILE *fr = fopen(fn, "rb");
    struct S buf2;
    size_t rc;
    while ((rc = fread(&buf2, sizeof(struct S), 1, fr)))
    {
        printf("a: %d, b: %d, s: %s\n", buf2.num, buf2.len, buf2.str);
    }

    fclose(fr);
    remove(fn);

    return 0;
}