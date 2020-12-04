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
#define MAX_ITER 1

int getChunkSize(int arrayCount, int numElPerLine);

int run(int run_number, int argc, char *argv[]) {
  int device_count, m_size;
  double cpu_perc;
  char *stats_output = nullptr;

  if (argc < 6) {
    m_size = 50;
    device_count = 1;
    cpu_perc = 0.1;
  } else {
    device_count = atoi(argv[2]);
    m_size = atoi(argv[1]);
    cpu_perc = atof(argv[3]);
    stats_output = argv[5];
  }
  if (cpu_perc > 1)
    cpu_perc /= 100;
  MatrixConfiguration conf = {device_count, m_size, m_size, cpu_perc};
  ExecutionStats stats;
  stats.id = conf.GetConfId();
  stats.total_jacobi_time = 0;
  stats.total_border_calc_time = 0;
  stats.total_inner_points_time = 0;
  stats.total_idle_comm_time = 0;
  stats.total_sweep_time = 0;
  stats.total_time_reducing_difference = 0;
  stats.total_time_waiting_to_host_transfer = 0;
  stats.total_time_waiting_to_device_transfer = 0;
  stats.last_global_difference = 0;
  stats.n_diff_reducions = 0;
  stats.n_sweeps = 0;
  printf("Run number: %d\n", run_number);
  printf("Configuration:%s\n", conf.GetConfId().c_str());
  double start = MPI_Wtime();
  pde_solver::Matrix m(conf);
  // old matrix
  m.Init(75, 100, 100, 0, 100, argc, argv);
  // new matrix. Stores the result of the Jacobi sweep.
  pde_solver::Matrix new_m = m.CloneShell();
  int num_iter = 0;
  float global_diff = 10;
  int process_id = GetProcessId();
  while (global_diff > EPSILON && num_iter < MAX_ITER) {
    float diff = m.LocalSweep(new_m, &stats);
    // printf("localdiff on proc %d: %f\n", process_id, diff);
    if (num_iter % 4 == 0) {
      global_diff = m.GlobalDifference(&stats);
      // printf("Global diff: %f\n", global_diff);
    }
    pde_solver::Matrix tmp = m;
    m = new_m;
    new_m = tmp;
    num_iter++;
    // m.ShowMatrix();
  }
  stats.total_jacobi_time = MPI_Wtime() - start;

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