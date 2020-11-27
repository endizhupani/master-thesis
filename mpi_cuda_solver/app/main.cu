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
#define MAX_ITER 10000

int getChunkSize(int arrayCount, int numElPerLine);

int run(int run_number, int argc, char *argv[]) {

  int deviceCount = 0;
#ifdef __CUDACC__

  CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
#endif // __CUDACC__
  char *file_name = argv[4];
  MatrixConfiguration conf;
  if (argc <= 2) {
    conf = {deviceCount, 100, 100, 0.25};
  } else {
    conf = {deviceCount, atoi(argv[1]), atoi(argv[1]), atof(argv[2])};
  }
  pde_solver::Matrix m(conf);
  printf("Initializing the matrix\n");
  m.Init(75, 100, 100, 0, 100, argc, argv);
  printf("Base matrix initialized. Cloning...\n");

  pde_solver::Matrix new_m = m.CloneShell();

  printf("Matrix cloned\n");
  int num_iter = 0;
  float global_diff = 10;
  ExecutionStats stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0};
  // m.ShowMatrix();
  // double tot_loop_time = 0;
  // double calc_start = MPI_Wtime();
  printf("Run number: %d\n", run_number);
  while (global_diff > EPSILON && num_iter < MAX_ITER) {

    // double t = MPI_Wtime();
    m.LocalSweep(new_m, &stats);
    // printf("Sweeped");
    // tot_loop_time += (MPI_Wtime() - t);
    if (num_iter % 4 == 0) {
      global_diff = m.GlobalDifference(&stats);
    }

    pde_solver::Matrix tmp = m;
    m = new_m;
    new_m = tmp;
    num_iter++;
  }

  stats.print_to_console();
  int process_id = GetProcessId();
  if (process_id == 0) {
    if (run_number == 1) {
      stats.PrintHeaderToFile(file_name);
    }

    stats.PrintToFile(file_name);
  }
  printf("\nFinished the computation. Deallocating...\n");
  new_m.Finalize();

  m.Finalize();
  m.FinalizeMpi();

  return 0;
}

int main(int argc,
         char *argv[]) { // matrix size, cpu percentage, numRuns, statsFile
  if (argc < 5) {
    printf("Please supply the matrix size, percentage to be calculated on the "
           "CPU, number of runs and the path for the statistics file.\n");
    exit(1);
  }

  int num_runs = atoi(argv[3]);
  InitMPIContext(argc, argv);
  for (int i = 1; i <= num_runs; i++) {
    run(i, argc, argv);
  }
  FinalizeMPIContext();
}

int getChunkSize(int arrayCount, int numElPerLine) {
  int num_threads = omp_get_num_threads();

  int chunk_size = ceil((double)arrayCount / num_threads);
  chunk_size += (chunk_size % numElPerLine);

  return chunk_size;
}