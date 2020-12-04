#include "cache_helpers.h"
#include "matrix.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace std;
#define N 30
#define EPSILON 0.01
#define MAX_ITER 5000
//#define CACHE_LINE_SIZE sysconf(_SC_LEVEL1_DCACHE_LINESIZE)

int getChunkSize(int arrayCount, int numElPerLine);

int main(int argc, char *argv[]) {
  // run_matrix_as_matrix();

  pde_solver::data::cpu_distr::Matrix m(1, N, N);
  m.Init(75, 100, 100, 0, 100, argc, argv);
  m.PrintMatrixInfo();
  pde_solver::data::cpu_distr::Matrix new_m = m.CloneShell();
  // m.PrintAllPartitions();
  // new_m.PrintAllPartitions();
  int num_iter = 0;
  double global_diff = 10;
  while (global_diff > EPSILON && num_iter < MAX_ITER) {

    m.LocalSweep(new_m);
    if (num_iter % 4 == 0) {
      global_diff = m.GlobalDifference();
    }

    pde_solver::data::cpu_distr::Matrix tmp = m;
    m = new_m;
    new_m = tmp;
    num_iter++;
  }

  m.Synchronize();
  m.PrintAllPartitions();
  // m.ShowMatrix();

  new_m.Deallocate();
  m.Synchronize();
  m.Finalize();
  printf("iter: %d\n", num_iter);
  printf("difference: %f\n", global_diff);
  return 0;
  // printf("======================================\n");
  // //run_matrix_as_array();
  // printf("Solved\n");
}

int getChunkSize(int arrayCount, int numElPerLine) {
  int num_threads = omp_get_num_threads();

  int chunk_size = ceil((double)arrayCount / num_threads);
  chunk_size += (chunk_size % numElPerLine);

  return chunk_size;
}