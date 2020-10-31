#include <iostream>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include "cache_helpers.h"
using namespace std;
#define EPSILON 0.01
#define MAX_ITER 5000

int getChunkSize(int arrayCount, int numElPerLine);
void run_matrix_as_array(int size_per_dim)
{
    int i, j;
    size_t cache_line_size_b = CacheLineSize();
    int dp_per_line = cache_line_size_b / sizeof(double);
    int max_threads = omp_get_max_threads();
    printf("Running on max %d threads.\nMatrix is stored as an array.\nMatrix size: %d rows by %d columns\nCache line size %zu\n", max_threads, size_per_dim, size_per_dim, cache_line_size_b);
    omp_set_num_threads(max_threads);
    double globalDiff = 500;
    double mean = 0.0;
    double *u = static_cast<double *>(aligned_alloc(cache_line_size_b, size_per_dim * size_per_dim * sizeof(double)));

    double *w = static_cast<double *>(aligned_alloc(cache_line_size_b, size_per_dim * size_per_dim * sizeof(double))); //new double[size_per_dim * size_per_dim];
    double *tmp;

#pragma omp parallel for schedule(static, getChunkSize(size_per_dim *size_per_dim, dp_per_line)) default(none) private(i) firstprivate(dp_per_line, size_per_dim) shared(u, w) reduction(+ \
                                                                                                                                                                                         : mean)
    for (i = 0; i < size_per_dim; i++)
    {
        u[i * size_per_dim] = u[i * size_per_dim + (size_per_dim - 1)] = u[i] = w[i * size_per_dim] = w[i * size_per_dim + (size_per_dim - 1)] = w[i] = 100;
        u[(size_per_dim - 1) * size_per_dim + i] = w[(size_per_dim - 1) * size_per_dim + i] = 0;
        mean += u[i * size_per_dim] + u[i * size_per_dim + (size_per_dim - 1)] + u[i] + u[(size_per_dim - 1) * size_per_dim + i];
    }

    mean /= (4 * size_per_dim);
#pragma omp parallel for schedule(static, getChunkSize(size_per_dim *size_per_dim, dp_per_line)) default(none) shared(u) private(i) firstprivate(mean, dp_per_line, size_per_dim)
    for (i = size_per_dim; i < (size_per_dim * size_per_dim) - size_per_dim; i++)
    {
        if (i % size_per_dim == 0 || (i - (size_per_dim - 1)) % size_per_dim == 0)
        {
            continue;
        }
        u[i] = mean;
    }

    double start = omp_get_wtime();
    int num_iter = 0;
    while (globalDiff > EPSILON && num_iter < MAX_ITER)
    {
        globalDiff = 0.0;

#pragma omp parallel for default(none) private(i) firstprivate(u, w, num_iter, dp_per_line, size_per_dim) reduction(max \
                                                                                                                    : globalDiff) schedule(static, getChunkSize(size_per_dim *size_per_dim, dp_per_line))

        for (i = size_per_dim + 1; i < (size_per_dim * size_per_dim) - size_per_dim - 1; i++)
        {
            if (i % size_per_dim == 0 || (i - (size_per_dim - 1)) % size_per_dim == 0)
            {
                continue;
            }
            auto previous = u[i];
            w[i] = (u[i - 1] + u[i + 1] + u[i - size_per_dim] + u[i + size_per_dim]) / 4;

            auto currentDifference = fabs(w[i] - previous);
            if (currentDifference > globalDiff)
            {
                globalDiff = currentDifference;
            }
        }

        if (globalDiff <= EPSILON)
            break;

        tmp = u;
        u = w;
        w = tmp;
        num_iter++;
    }

    double end = omp_get_wtime();
    printf("time ellapsed = %f\n", end - start);
    printf("num_iter %d\n", num_iter);
    // for (int i = 0; i < size_per_dim; i++)
    // {
    //     for (int j = 0; j < size_per_dim; j++)
    //     {
    //         printf("%6.2f ", u[i * size_per_dim + j]);
    //     }
    //     putchar('\n');
    // }
    // delete[] u;
    // delete[] w;
    free(w);
    free(u);
}

void report_num_threads(int level)
{
    printf("Level %d: hello form thread nr: %d\n",
           level, omp_get_thread_num());
#pragma omp single
    {
        printf("Level %d: number of threads in the team - %d\n",
               level, omp_get_num_threads());
    }
}

int main(int argc, char *argv[])
{
    // if (argc <= 1)
    // {
    //     printf("Please enter the size of the matrix\n");
    //     return 1;
    // }
    // //run_matrix_as_matrix();

    // //printf("======================================\n");
    // run_matrix_as_array(atoi(argv[1]));
    omp_set_nested(2);
#pragma omp parallel num_threads(2)
    {
        report_num_threads(1);

#pragma omp parallel num_threads(4)
        {
            report_num_threads(2);
        }
    }
}

int getChunkSize(int arrayCount, int numElPerLine)
{
    int num_threads = omp_get_num_threads();

    int chunk_size = ceil((double)arrayCount / num_threads);
    chunk_size += (chunk_size % numElPerLine);

    return chunk_size;
}