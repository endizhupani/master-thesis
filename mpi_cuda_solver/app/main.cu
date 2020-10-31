#include "cache_helpers.h"
//#include "hemi.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <matrix.h>
#include <omp.h>

using namespace std;
#define N 1000
#define EPSILON 0.01
#define N_THREADS 8
#define MAX_ITER 5000

int getChunkSize(int arrayCount, int numElPerLine);

int main(int argc,
         char *argv[]) { // matrix size, num threads per process, cpu percentage

#ifdef __CUDACC__
  int deviceCount = 0;
  CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));

  printf("Number of GPUs: %d\n", deviceCount);
  printf("works\n");
#endif // __CUDACC__

  pde_solver::Matrix m(1, N, N);
  m.Init(75, 100, 100, 0, 100, argc, argv);
  m.PrintMatrixInfo();
  pde_solver::Matrix new_m = m.CloneShell();
  int num_iter = 0;
  double global_diff = 10;
  double tot_loop_time = 0;
  double calc_start = MPI_Wtime();
  while (global_diff > EPSILON && num_iter < MAX_ITER) {

    double t = MPI_Wtime();
    m.LocalSweep(new_m);
    tot_loop_time += (MPI_Wtime() - t);
    if (num_iter % 4 == 0) {
      global_diff = m.GlobalDifference();
    }

    pde_solver::Matrix tmp = m;
    m = new_m;
    new_m = tmp;
    num_iter++;
  }
  double calc_time = MPI_Wtime() - calc_start;

  printf("AVG sweep time: %f\n", (tot_loop_time / num_iter));
  printf("Total Calculaiton time: %f\n", calc_time);

  // m.Synchronize();
  // m.PrintAllPartitions();
  // m.ShowMatrix();

  new_m.Deallocate();

  m.Finalize();
  printf("iter: %d\n", num_iter);
  printf("difference: %f\n", global_diff);
  return 0;
}

int getChunkSize(int arrayCount, int numElPerLine) {
  int num_threads = omp_get_num_threads();

  int chunk_size = ceil((double)arrayCount / num_threads);
  chunk_size += (chunk_size % numElPerLine);

  return chunk_size;
}