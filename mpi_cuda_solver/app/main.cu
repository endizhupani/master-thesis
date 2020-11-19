#include "cache_helpers.h"
#include "common.h"
//#include "hemi.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <matrix.h>
#include <omp.h>

using namespace std;
#define EPSILON 0.01
#define N_THREADS 8
#define MAX_ITER 5000

int getChunkSize(int arrayCount, int numElPerLine);

int main(int argc,
         char *argv[]) { // matrix size, cpu percentage

  int deviceCount = 0;
#ifdef __CUDACC__

  CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
#endif // __CUDACC__
  MatrixConfiguration conf;
  if (argc <= 2) {
    conf = {deviceCount, 100, 100, 0.25};
  } else {
    conf = {deviceCount, atoi(argv[1]), atoi(argv[1]), atof(argv[2])};
  }
  pde_solver::Matrix m(conf);
  m.Init(75, 100, 100, 0, 100, argc, argv);
  pde_solver::Matrix new_m = m.CloneShell();
  int num_iter = 0;
  float global_diff = 10;
  ExecutionStats stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0};
  // m.ShowMatrix();
  // double tot_loop_time = 0;
  // double calc_start = MPI_Wtime();
  printf("Starting Jacobi Calculation\n");
  while (global_diff > EPSILON && num_iter < MAX_ITER) {

    // double t = MPI_Wtime();
    m.LocalSweep(new_m, &stats);
    // tot_loop_time += (MPI_Wtime() - t);
    if (num_iter % 4 == 0) {
      global_diff = m.GlobalDifference(&stats);
    }

    pde_solver::Matrix tmp = m;
    m = new_m;
    new_m = tmp;
    num_iter++;
  }

  // m.ShowMatrix();
  stats.print_to_console();

  // double calc_time = MPI_Wtime() - calc_start;

  // printf("AVG sweep time: %f\n", (tot_loop_time / num_iter));
  // printf("Total Calculaiton time: %f\n", calc_time);

  // m.Synchronize();
  // m.PrintAllPartitions();
  //
  printf("\nFinished the computation. Deallocating...\n");
  new_m.Deallocate();

  m.Finalize();
  return 0;
}

int getChunkSize(int arrayCount, int numElPerLine) {
  int num_threads = omp_get_num_threads();

  int chunk_size = ceil((double)arrayCount / num_threads);
  chunk_size += (chunk_size % numElPerLine);

  return chunk_size;
}