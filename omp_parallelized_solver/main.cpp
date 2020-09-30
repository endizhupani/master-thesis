#include <iostream>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include "cache_helpers.h"
using namespace std;
#define N 4000
#define EPSILON 0.01
#define N_THREADS 8
#define MAX_ITER 5000

int getChunkSize(int arrayCount, int numElPerLine);

void run_matrix_as_matrix()
{
    printf("Running on max %d threads.\nMatrix is stored as a matrix.\nMatrix size: %d rows by %d columns\n", N_THREADS, N, N);
    omp_set_num_threads(N_THREADS);

    double globalDiff = 500;
    double mean = 0.0, start, end;
    auto u = new double[N][N];
    auto w = new double[N][N];
    double *tmp;

#pragma omp parallel for default(none) shared(u, w) schedule(static) reduction(+ \
                                                                               : mean)
    for (int i = 0; i < N; i++)
    {
        u[i][0] = w[i][0] = u[i][N - 1] = w[i][N - 1] = u[0][i] = w[0][i] = 100;
        u[N - 1][i] = w[N - 1][i] = 0;

        mean += u[i][0] + u[i][N - 1] + u[0][i] + u[N - 1][i];
    }

    mean /= (4 * N);

#pragma omp parallel for default(none) shared(u) firstprivate(mean) schedule(static)
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            u[i][j] = mean;
        }
    }

    start = omp_get_wtime();
    int num_iter = 0;

    while (globalDiff > EPSILON && num_iter < MAX_ITER)
    {
        globalDiff = 0.0;

#pragma omp parallel for reduction(max \
                                   : globalDiff) schedule(static)
        for (int i = 1; i < N - 1; i++)
        {
#pragma omp simd
            for (int j = 1; j < N - 1; j++)
            {
                w[i][j] = 1; //(u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4;
                auto diff = fabs(w[i][j] - u[i][j]);
                if (diff > globalDiff)
                {
                    globalDiff = diff;
                }
            }
        }

        swap(w, u);
        num_iter++;
    }

    end = omp_get_wtime();
    // for (int i = N - 2; i < N - 1; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%6.2f ", u[i][j]);
    //     }
    //     putchar('\n');
    // }
    printf("time ellapsed = %f\n", end - start);
    printf("num_iter %d\n", num_iter);
    delete[] u;
    delete[] w;
}
void run_matrix_as_array()
{
    int i, j;
    size_t cache_line_size_b = CacheLineSize();
    int dp_per_line = cache_line_size_b / sizeof(double);
    int max_threads = omp_get_max_threads();
    printf("Running on max %d threads.\nMatrix is stored as an array.\nMatrix size: %d rows by %d columns\nCache line size %zu\n", max_threads, N, N, cache_line_size_b);
    omp_set_num_threads(max_threads);
    double globalDiff = 500;
    double mean = 0.0;
    double *u = static_cast<double *>(aligned_alloc(cache_line_size_b, N * N * sizeof(double)));

    double *w = static_cast<double *>(aligned_alloc(cache_line_size_b, N * N * sizeof(double))); //new double[N * N];
    double *tmp;

#pragma omp parallel for schedule(static, getChunkSize(N *N, dp_per_line)) default(none) private(i) firstprivate(dp_per_line) shared(u, w) reduction(+ \
                                                                                                                                                     : mean)
    for (i = 0; i < N; i++)
    {
        u[i * N] = u[i * N + (N - 1)] = u[i] = w[i * N] = w[i * N + (N - 1)] = w[i] = 100;
        u[(N - 1) * N + i] = w[(N - 1) * N + i] = 0;
        mean += u[i * N] + u[i * N + (N - 1)] + u[i] + u[(N - 1) * N + i];
    }

    mean /= (4 * N);
#pragma omp parallel for schedule(static, getChunkSize(N *N, dp_per_line)) default(none) shared(u) private(i) firstprivate(mean, dp_per_line)
    for (i = N; i < (N * N) - N; i++)
    {
        if (i % N == 0 || (i - (N - 1)) % N == 0)
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

#pragma omp parallel for default(none) private(i) firstprivate(u, w, num_iter, dp_per_line) reduction(max \
                                                                                                      : globalDiff) schedule(static, getChunkSize(N *N, dp_per_line))

        for (i = N + 1; i < (N * N) - N - 1; i++)
        {
            if (i % N == 0 || (i - (N - 1)) % N == 0)
            {
                continue;
            }
            auto previous = u[i];
            w[i] = (u[i - 1] + u[i + 1] + u[i - N] + u[i + N]) / 4;

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
    //    for(int i = 0; i < N; i++){
    //        for (int j = 0; j < N; j++) {
    //            printf("%6.2f ", u[i*N + j]);
    //        }
    //        putchar('\n');
    //    }
    // delete[] u;
    // delete[] w;
    free(w);
    free(u);
}

int main()
{
    //run_matrix_as_matrix();

    //printf("======================================\n");
    run_matrix_as_array();
}

int getChunkSize(int arrayCount, int numElPerLine)
{
    int num_threads = omp_get_num_threads();

    int chunk_size = ceil((double)arrayCount / num_threads);
    chunk_size += (chunk_size % numElPerLine);

    return chunk_size;
}