#include <iostream>
#include <math.h>
#include <chrono>

#define N 10000
#define EPSILON 0.01

int main() {
    double diff, mean;
    int i, j;
    double **u = new double*[N];
    for (i = 0; i < N; i++) {
        u[i] = new double[N];
    }

    double **w = new double*[N];
    for (i = 0; i < N; i++) {
        w[i] = new double[N];
    }

    mean = 0,0;
    for (i = 0; i < N; i++) {
        u[i][0] = u[i][N-1] = u[0][i] = 100;
        u[N-1][i] = 0;
        mean += u[i][0] + u[i][N-1] + u[0][i] + u[N-1][i];
    }

    mean /= (4 * N);

    for(i = 1; i < N-1; i++){
        for (j = 1; j < N-1; j++) {
            u[i][j] = mean;
        }
    }

    auto full_calc_start = std::chrono::high_resolution_clock::now();
    int num_iter = 0;
    double calc_loop_min = 1000000;
    double calc_loop_max = 0;
    double calc_loop_avg = 0;
    double val_assign_loop_min = 1000000;
    double val_assign_loop_max = 0;
    double val_assign_loop_avg = 0;
    for (;;) {
        num_iter++;
        diff = 0,0;

        auto calc_loop_start = std::chrono::high_resolution_clock::now();
        for (i = 1; i < N-1; i++) {
            for (j = 1; j < N-1; j++) {
                w[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])/4;

                if (fabs(w[i][j] - u[i][j]) > diff){
                    diff = fabs(w[i][j] - u[i][j]);
                }
            }
        }
        auto calc_loop_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> calc_loop_elapsed = calc_loop_end - calc_loop_start;
        if (calc_loop_elapsed.count() > calc_loop_max){
            calc_loop_max = calc_loop_elapsed.count();
        }

        if (calc_loop_elapsed.count() < calc_loop_min){
            calc_loop_min = calc_loop_elapsed.count();
        }

        calc_loop_avg += calc_loop_elapsed.count();


        if (diff <= EPSILON) break;

        auto val_assign_loop_start = std::chrono::high_resolution_clock::now();
        for(i = 1; i < N-1; i++){
            for (j = 1; j < N-1; j++) {
                u[i][j] = w[i][j];
            }
        }
        auto val_assign_loop_finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> val_assign_loop_elapsed = val_assign_loop_finish - val_assign_loop_start;
        if (val_assign_loop_elapsed.count() > val_assign_loop_max){
            val_assign_loop_max = val_assign_loop_elapsed.count();
        }

        if (val_assign_loop_elapsed.count() < val_assign_loop_min){
            val_assign_loop_min = val_assign_loop_elapsed.count();
        }

        val_assign_loop_avg += val_assign_loop_elapsed.count();
    }

    auto full_calc_finish = std::chrono::high_resolution_clock::now();
    printf("num_iter: %d\n", num_iter);
    std::chrono::duration<double> elapsed =full_calc_finish - full_calc_start;
    printf("full_calc_time: %f\n", elapsed.count());

    printf("calc_loop_min: %f\n", calc_loop_min);
    printf("calc_loop_max: %f\n", calc_loop_max);
    printf("calc_loop_avg: %f\n", calc_loop_avg / num_iter);

    printf("val_assign_loop_min: %f\n", val_assign_loop_min);
    printf("val_assign_loop_max: %f\n", val_assign_loop_max);
    printf("val_assign_loop_avg: %f\n", val_assign_loop_avg / (num_iter-1));

//        for(i = 0; i < N; i++){
//        for (j = 0; j < N; j++) {
//            printf("%6.2f ", u[i][j]);
//        }
//        putchar('\n');
//    }

    for (i = 0; i < N; i++) {
        delete[] u[i];
    }

    delete [] u;

    for (i = 0; i < N; i++) {
        delete [] w[i];
    }

    delete [] w;

printf("Solved\n");
}
