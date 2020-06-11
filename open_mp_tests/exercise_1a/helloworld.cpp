#include <stdio.h>
#include <omp.h>
int main()
{
#pragma omp parallel
    {
        printf("hello(%d) ", omp_get_thread_num());
        printf("world(%d) \n", omp_get_thread_num());
    }
    return 0;
}