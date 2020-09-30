#include <iostream>
#include <math.h>
#include <chrono>
#include <array>
using namespace std;
#define N 4000
#define EPSILON 0.01
#define MAX_ITER 10000

void run_matrix_as_array()
{
    printf("Matrix is stored as an array.\nMatrix size: %d rows by %d columns\n", N, N);
    int i, j;
    double globalDiff = 500;
    double mean = 0.0;
    double *u = new double[N * N];
    double *w = new double[N * N];
    double *tmp;
    for (i = 0; i < N; i++)
    {
        u[i * N] = u[i * N + (N - 1)] = u[i] = w[i * N] = w[i * N + (N - 1)] = w[i] = 100;
        u[(N - 1) * N + i] = w[(N - 1) * N + i] = 0;
        mean += u[i * N] + u[i * N + (N - 1)] + u[i] + u[(N - 1) * N + i];
    }

    mean /= (4 * N);

    for (i = N; i < (N * N) - N; i++)
    {
        if (i % N == 0 || (i - (N - 1)) % N == 0)
        {
            continue;
        }
        u[i] = mean;
    }

    auto full_calc_start = std::chrono::high_resolution_clock::now();
    int num_iter = 0;

    while (globalDiff > EPSILON && num_iter < MAX_ITER)
    {
        globalDiff = 0.0;

        auto calc_loop_start = std::chrono::high_resolution_clock::now();
        for (i = N + 1; i < (N * N) - N - 1; i++)
        {
            if (i % N == 0 || (i - (N - 1)) % N == 0)
            {
                continue;
            }

            w[i] = (u[i - 1] + u[i + 1] + u[i - N] + u[i + N]) / 4;
            auto difference = fabs(w[i] - u[i]);
            if (difference > globalDiff)
            {
                globalDiff = difference;
            }
        }

        num_iter++;
        if (globalDiff <= EPSILON || num_iter > 1815)
            break;

        tmp = u;
        u = w;
        w = tmp;
    }

    auto full_calc_finish = std::chrono::high_resolution_clock::now();
    printf("Results with matrix stored as row major array: \n");
    printf("num_iter: %d\n", num_iter);
    std::chrono::duration<double> elapsed = full_calc_finish - full_calc_start;
    printf("full_calc_time: %f\n", elapsed.count());

    delete[] u;
    delete[] w;
}

int main()
{
    run_matrix_as_array();
}
