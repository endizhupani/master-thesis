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
  int device_count, m_size;
  float cpu_perc;
  char *stats_output = nullptr;

  if (argc < 6) {
    m_size = 10;
    device_count = 1;
    cpu_perc = 0.2;
  } else {
    device_count = atoi(argv[2]);
    m_size = atoi(argv[1]);
    cpu_perc = atof(argv[3]);
    if (cpu_perc > 1)
      cpu_perc /= cpu_perc;
    *stats_output = argv[5];
  }

  MatrixConfiguration conf = {device_count, m_size, m_size, cpu_perc};
  ExecutionStats stats = {
      conf.GetConfId(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, , 0.0, 0, 0};
  printf("Run number: %d\n", run_number);
  double start = MPI_Wtime();
  pde_solver::Matrix m(conf);
  printf("Initializing the matrix\n");

  // old matrix
  m.Init(75, 100, 100, 0, 100, argc, argv);
  printf("Base matrix initialized. Cloning...\n");

  // new matrix. Stores the result of the Jacobi sweep.
  pde_solver::Matrix new_m = m.CloneShell();
  int num_iter = 0;
  float global_diff = 10;

  while (global_diff > EPSILON && num_iter < MAX_ITER) {
    m.LocalSweep(new_m, &stats);
    if (num_iter % 4 == 0) {
      global_diff = m.GlobalDifference(&stats);
    }

    pde_solver::Matrix tmp = m;
    m = new_m;
    new_m = tmp;
    num_iter++;
  }
  stats.total_jacobi_time = start - MPI_Wtime();

  int process_id = GetProcessId();
  if (process_id == 0) {
    if (stats_output) {
      if (run_number == 1) {
        stats.PrintHeaderToFile(stats_output);
      }
      stats.PrintToFile(stats_output);
    } else {
      stats.print_to_console();
    }
  }
  printf("\nFinished the computation. Deallocating...\n");
  new_m.Finalize();

  m.Finalize();
  m.FinalizeMpi();

  return 0;
}

int main(int argc,
         char *argv[]) { // matrix size, cpu percentage, numRuns, statsFile
  int num_runs;
  if (argc < 6) {
    printf("Please supply the matrix size, number of GPUs, percentage to be "
           "calculated on the "
           "CPU, number of runs and the path for the statistics file.\n");
    num_runs = 1;
  } else {
    num_runs = atoi(argv[4]);
  }
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