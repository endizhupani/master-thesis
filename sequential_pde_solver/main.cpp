#include <iostream>
#include <math.h>

#define N 50
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

    for (;;) {
        diff = 0,0;
        for (i = 1; i < N-1; i++) {
            for (j = 1; j < N-1; j++) {
                w[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])/4;

                if (fabs(w[i][j] - u[i][j]) > diff){
                    diff = fabs(w[i][j] - u[i][j]);
                }
            }
        }

        if (diff <= EPSILON) break;
        for(i = 1; i < N-1; i++){
            for (j = 1; j < N-1; j++) {
                u[i][j] = w[i][j];
            }
        }
    }

        for(i = 0; i < N; i++){
        for (j = 0; j < N; j++) {
            printf("%6.2f ", u[i][j]);
        }
        putchar('\n');
    }

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
