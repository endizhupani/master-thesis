#include <iostream>
#include <math.h>
#include <chrono>
#include <array>
using namespace std;
#define N 4000
#define EPSILON 0.01

void run_matrix_as_array()
{

    std::array<int, N> myArray;
    for (int i = 0; i < N; i++)
    {
        myArray[i] = i;
    }

    return;
    int i, j;
    double globalDiff;
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
    double calc_loop_min = 1000000;
    double calc_loop_max = 0;
    double calc_loop_avg = 0;

    for (;;)
    {
        num_iter++;
        globalDiff = 0.0;

        auto calc_loop_start = std::chrono::high_resolution_clock::now();
        for (i = N + 1; i < (N * N) - N - 1; i++)
        {
            if (i % N == 0 || (i - (N - 1)) % N == 0)
            {
                continue;
            }

            // w[i] = (u[i - 1] + u[i + 1] + u[i - N] + u[i + N]) / 4;
            // if (fabs(w[i] - u[i]) > globalDiff)
            // {
            //     globalDiff = fabs(w[i] - u[i]);
            // }
            w[i] = u[i];
        }
        auto calc_loop_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> calc_loop_elapsed = calc_loop_end - calc_loop_start;
        if (calc_loop_elapsed.count() > calc_loop_max)
        {
            calc_loop_max = calc_loop_elapsed.count();
        }

        if (calc_loop_elapsed.count() < calc_loop_min)
        {
            calc_loop_min = calc_loop_elapsed.count();
        }

        calc_loop_avg += calc_loop_elapsed.count();

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

    printf("calc_loop_min: %f\n", calc_loop_min);
    printf("calc_loop_max: %f\n", calc_loop_max);
    printf("calc_loop_avg: %f\n", calc_loop_avg / num_iter);

    //        for(i = 0; i < N; i++){
    //        for (j = 0; j < N; j++) {
    //            printf("%6.2f ", u[i][j]);
    //        }
    //        putchar('\n');
    //    }

    delete[] u;
    delete[] w;
}

void run_matrix_as_matrix()
{

    double globalDiff = 500, mean;
    int i, j;
    auto u = new double[N][N];
    auto w = new double[N][N];

    mean = 0.0;
    for (i = 0; i < N; i++)
    {
        u[i][0] = w[i][0] = u[i][N - 1] = w[i][N - 1] = u[0][i] = w[0][i] = 100;
        u[N - 1][i] = w[N - 1][i] = 0;
        mean += u[i][0] + u[i][N - 1] + u[0][i] + u[N - 1][i];
    }

    mean /= (4 * N);

    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            u[i][j] = mean;
        }
    }
    auto full_calc_start = std::chrono::high_resolution_clock::now();
    int num_iter = 0;
    double calc_loop_min = 1000000;
    double calc_loop_max = 0;
    double calc_loop_avg = 0;
    while (globalDiff > EPSILON && num_iter < 1)
    {
        globalDiff = 0.0;

        for (int i = 1; i < N - 1; i++)
        {

            for (int j = 1; j < N - 1; j++)
            {
                w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4;
                globalDiff = max(globalDiff, fabs(w[i][j] - u[i][j]));
            }
        }

        swap(w, u);

        if (num_iter++ % 100 == 0)
            printf("%5d, %0.6f\n", num_iter, globalDiff);
    }

    auto full_calc_finish = std::chrono::high_resolution_clock::now();
    printf("Results with matrix stored as array of pointers: \n");
    printf("num_iter: %d\n", num_iter);
    std::chrono::duration<double> elapsed = full_calc_finish - full_calc_start;
    printf("full_calc_time: %f\n", elapsed.count());

    delete[] u;

    delete[] w;
}

int main()
{
    run_matrix_as_matrix();
    //run_matrix_as_array();
}
