#include <stdio.h>
#include <omp.h>
static long num_steps = 1000000000;
double step;
#define NTHREADS 8
int main()
{
    double start = omp_get_wtime();
    int i, nthreads;
    double pi, x;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NTHREADS);
    printf("System has %d processors\n", omp_get_num_procs());

#pragma omp parallel for private(x) reduction(+ \
                                              : pi)
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        pi += 4.0 / (1.0 + x * x);
    }
    pi *= step;

    printf("pi = %f\n", pi);
    printf("time ellapsed = %f\n", omp_get_wtime() - start);
}