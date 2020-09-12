#include <iostream>
#include <math.h>
#include <omp.h>
using namespace std;
#define N 10000
#define EPSILON 0.01
#define N_THREADS 2

int getChunkSize(int arraySize);

void run_matrix_as_matrix()
{
    int i, j;
    printf("Running on max %d threads.\nMatrix is stored as a matrix.\nMatrix size: %d rows by %d columns\n", N_THREADS, N, N);
    omp_set_num_threads(N_THREADS);

    double globalDiff;
    double mean = 0.0;
    auto u = new double[N][N];
    auto w = new double[N][N];
    double *tmp;

#pragma omp parallel default(none) private(i) shared(u, w) reduction(+ \
                                                                     : mean)
    {
#pragma omp for schedule(static)
        for (i = 0; i < N; i++)
        {
            u[i][0] = w[i][0] = u[i][N - 1] = w[i][N - 1] = u[0][i] = w[0][i] = 100;
            u[N - 1][i] = w[N - 1][i] = 0;

            mean += u[i][0] + u[i][N - 1] + u[0][i] + u[N - 1][i];
        }
    }
    mean /= (4 * N);
#pragma omp parallel default(none) shared(u) private(i, j) firstprivate(mean)
    {
#pragma omp for schedule(static)
        for (i = 1; i < N - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                u[i][j] = mean;
            }
        }
    }

    double calc_loop_min = 1000000;
    double calc_loop_max = 0;
    double calc_loop_avg = 0;

    double start = omp_get_wtime();
    int num_iter = 0;
    for (;;)
    {
        num_iter++;
        globalDiff = 0.0;
        double calc_loop_start = omp_get_wtime();
#pragma omp parallel default(none) private(i, j) firstprivate(u, w) reduction(max \
                                                                              : globalDiff)
        {
#pragma omp for schedule(static)
            for (i = 1; i < N - 1; i++)
            {
                for (j = 1; j < N - 1; j++)
                {
                    // w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4;

                    // if (fabs(w[i][j] - u[i][j]) > globalDiff)
                    // {
                    //     globalDiff = fabs(w[i][j] - u[i][j]);
                    // }
                    w[i][j] = u[i][j];
                }
            }
        }
        double calc_loop_dur = omp_get_wtime() - calc_loop_start;
        calc_loop_avg += calc_loop_dur;
        if (calc_loop_dur < calc_loop_min)
        {
            calc_loop_min = calc_loop_dur;
        }

        if (calc_loop_dur > calc_loop_max)
        {
            calc_loop_max = calc_loop_dur;
        }

        if (globalDiff <= EPSILON || num_iter > 1815)
            break;
        swap(w, u);
        // #pragma omp parallel default(none) private(i, j) firstprivate(u, w)
        //         {
        // #pragma omp for schedule(static)
        //             for (i = 1; i < N - 1; i++)
        //             {
        //                 for (j = 1; j < N - 1; j++)
        //                 {
        //                     u[i][j] = w[i][j];
        //                 }
        //             }
        //         }
    }

    double end = omp_get_wtime();
    printf("time ellapsed = %f\n", end - start);
    printf("num_iter %d\n", num_iter);
    printf("calc_loop_min: %f\n", calc_loop_min);
    printf("calc_loop_max: %f\n", calc_loop_max);
    printf("calc_loop_avg: %f\n", calc_loop_avg / num_iter);

    delete[] u;
    delete[] w;
}
void run_matrix_as_array()
{
    int i, j;
    printf("Running on max %d threads.\nMatrix is stored as an array.\nMatrix size: %d rows by %d columns\n", N_THREADS, N, N);
    omp_set_num_threads(N_THREADS);
    double globalDiff;
    double mean = 0.0;
    double *u = new double[N * N];
    double *w = new double[N * N];
    double *tmp;

#pragma omp parallel default(none) private(i) shared(u, w) reduction(+ \
                                                                     : mean)
    {
#pragma omp for schedule(static, getChunkSize(N *N))
        for (i = 0; i < N; i++)
        {
            u[i * N] = u[i * N + (N - 1)] = u[i] = w[i * N] = w[i * N + (N - 1)] = w[i] = 100;
            u[(N - 1) * N + i] = w[(N - 1) * N + i] = 0;
            mean += u[i * N] + u[i * N + (N - 1)] + u[i] + u[(N - 1) * N + i];
        }
    }
    mean /= (4 * N);
#pragma omp parallel default(none) shared(u) private(i) firstprivate(mean)
    {
#pragma omp for schedule(static, getChunkSize(N *N))
        for (i = N; i < (N * N) - N; i++)
        {
            if (i % N == 0 || (i - (N - 1)) % N == 0)
            {
                continue;
            }
            u[i] = mean;
        }
    }

    double calc_loop_min = 1000000;
    double calc_loop_max = 0;
    double calc_loop_avg = 0;

    double start = omp_get_wtime();
    int num_iter = 0;
    for (;;)
    {
        num_iter++;
        globalDiff = 0.0;
        double calc_loop_start = omp_get_wtime();
#pragma omp parallel default(none) private(i) firstprivate(u, w, num_iter) reduction(max \
                                                                                     : globalDiff)
        {
#pragma omp for schedule(static, getChunkSize(N *N))
            for (i = N + 1; i < (N * N) - N - 1; i++)
            {
                if (i % N == 0 || (i - (N - 1)) % N == 0)
                {
                    continue;
                }

                w[i] = (u[i - 1] + u[i + 1] + u[i - N] + u[i + N]) / 4;
                if (fabs(w[i] - u[i]) > globalDiff)
                {
                    globalDiff = fabs(w[i] - u[i]);
                }
            }
        }
        double calc_loop_dur = omp_get_wtime() - calc_loop_start;
        calc_loop_avg += calc_loop_dur;
        if (calc_loop_dur < calc_loop_min)
        {
            calc_loop_min = calc_loop_dur;
        }

        if (calc_loop_dur > calc_loop_max)
        {
            calc_loop_max = calc_loop_dur;
        }

        if (globalDiff <= EPSILON)
            break;

        tmp = u;
        u = w;
        w = tmp;
    }

    double end = omp_get_wtime();
    printf("time ellapsed = %f\n", end - start);
    printf("num_iter %d\n", num_iter);
    printf("calc_loop_min: %f\n", calc_loop_min);
    printf("calc_loop_max: %f\n", calc_loop_max);
    printf("calc_loop_avg: %f\n", calc_loop_avg / num_iter);
    //    for(int i = 0; i < N; i++){
    //        for (int j = 0; j < N; j++) {
    //            printf("%6.2f ", u[i*N + j]);
    //        }
    //        putchar('\n');
    //    }
    delete[] u;
    delete[] w;
}

int main()
{
    run_matrix_as_matrix();
    printf("======================================\n");
    //run_matrix_as_array();
    printf("Solved\n");
}

int getChunkSize(int arraySize)
{
    int num_threads = omp_get_num_threads();
    return ceil((double)arraySize / num_threads);
}