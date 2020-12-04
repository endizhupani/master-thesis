#include "cache_helpers.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>

#define EPSILON 0.01
#define MAX_ITER 5000

int getChunkSize(int arrayCount, int numElPerLine);
double run(int size_per_dim) {

  size_t cache_line_size_b = CacheLineSize();
  int dp_per_line = cache_line_size_b / sizeof(float);
  int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  float globalDiff = 500;
  float *u = static_cast<float *>(aligned_alloc(
      cache_line_size_b, size_per_dim * size_per_dim * sizeof(float)));

  float *w = static_cast<float *>(aligned_alloc(
      cache_line_size_b,
      size_per_dim * size_per_dim *
          sizeof(float))); // new double[size_per_dim * size_per_dim];
  float *tmp;

#pragma omp parallel for schedule(                                             \
    static, getChunkSize(size_per_dim *size_per_dim, dp_per_line))             \
    firstprivate(dp_per_line, size_per_dim)
  for (int i = 0; i < (size_per_dim * size_per_dim); i++) {
    u[i] = 75;
  }

  int num_iter = 0;
  double start = omp_get_wtime();
  while (globalDiff > EPSILON && num_iter < MAX_ITER) {
    globalDiff = 0.0;

#pragma omp parallel for firstprivate(dp_per_line,                             \
                                      size_per_dim) reduction(max              \
                                                              : globalDiff)    \
    schedule(static, getChunkSize(size_per_dim *size_per_dim, dp_per_line))

    for (int i = 0; i < (size_per_dim * size_per_dim); i++) {
      int row, col;
      row = i / size_per_dim;
      col = i % size_per_dim;
      float top, bottom, right, left;
      if (row == 0) {
        top = 100;
      } else {
        top = u[i - size_per_dim];
      }
      if (row == size_per_dim - 1) {
        bottom = 0;
      } else {
        bottom = u[i + size_per_dim];
      }

      if (col == 0) {
        left = 100;
      } else {
        left = u[i - 1];
      }
      float previous = u[i]; // place here because chances are it will be
                             // fetched from the cache
      if (col == size_per_dim - 1) {
        right = 100;
      } else {
        right = u[i + 1];
      }

      w[i] = (top + bottom + right + left) / 4;

      float currentDifference = fabs(w[i] - previous);
      if (currentDifference > globalDiff) {
        globalDiff = currentDifference;
      }
    }

    tmp = u;
    u = w;
    w = tmp;
    num_iter++;
  }

  double end = omp_get_wtime();
  // for (int i = 0; i < size_per_dim; i++) {
  //   for (int j = 0; j < size_per_dim; j++) {
  //     printf("%6.2f ", u[i * size_per_dim + j]);
  //   }
  //   putchar('\n');
  // }
  std::cout << "Done in: " << num_iter << " iterations" << std::endl;
  free(w);
  free(u);
  return end - start;
}

void report_num_threads(int level) {
  printf("Level %d: hello form thread nr: %d\n", level, omp_get_thread_num());
#pragma omp single
  {
    printf("Level %d: number of threads in the team - %d\n", level,
           omp_get_num_threads());
  }
}

int main(int argc, char *argv[]) {
  int m_size = 1000;
  int n_runs = 1;
  char *file;
  if (argc < 4) {
    std::cout << "please specify the matrix size, num runs and the stats "
                 "output file "
              << std::endl;
  } else {
    m_size = atoi(argv[1]);
    n_runs = atoi(argv[2]);
    file = argv[3];
  }
  double tot_time = 0;
  for (int i = 1; i <= n_runs; i++) {
    tot_time += run(m_size);
  }

  double avg = tot_time / n_runs;
  if (!file) {
    std::cout << "Average time =" << avg << std::endl;
    exit(0);
  }
  std::ifstream f(file);
  bool is_empty = f.peek() == std::ifstream::traits_type::eof();
  f.close();
  std::ofstream outputFile;
  outputFile.open(file, std::ios_base::app);
  if (is_empty) {
    outputFile << "id,avg_time\n";
  }
  outputFile << m_size << "," << avg << "\n";
  outputFile.close();
  exit(0);
}

int getChunkSize(int arrayCount, int numElPerLine) {
  int num_threads = omp_get_num_threads();

  int chunk_size = ceil((double)arrayCount / num_threads);
  chunk_size += (chunk_size % numElPerLine);

  return chunk_size;
}