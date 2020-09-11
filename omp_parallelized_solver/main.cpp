#include <iostream>
#include <math.h>
#include <omp.h>

#define N 1000
#define EPSILON 0.01

int getChunkSize(int arraySize);

int main()
{
    int n_threads = 1, i, j;
    printf("Running on max %d threads.\n", n_threads);
    omp_set_num_threads(n_threads);
    double mean;
    double *u = new double[N * N];
    double *w = new double[N * N];
    double *tmp;

    mean = 0, 0;

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
        double globalDiff = 0.0;
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

    printf("Solved\n");
}

int getChunkSize(int arraySize)
{
    int num_threads = omp_get_num_threads();
    return ceil((double)arraySize / num_threads);
}