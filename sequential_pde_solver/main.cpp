#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;
#define EPSILON 0.01
#define MAX_ITER 10000

double run_matrix_as_array(int matrix_size) {
  auto full_calc_start = std::chrono::high_resolution_clock::now();
  printf("Matrix is stored as an array.\nMatrix size: %d rows by %d columns\n",
         matrix_size, matrix_size);
  int i, j;
  double globalDiff = 500;
  double *u = new double[matrix_size * matrix_size];
  double *w = new double[matrix_size * matrix_size];
  double *tmp;
  for (i = 0; i < matrix_size; i++) {
    u[i * matrix_size] = u[i * matrix_size + (matrix_size - 1)] = u[i] =
        w[i * matrix_size] = w[i * matrix_size + (matrix_size - 1)] = w[i] =
            100;
    u[(matrix_size - 1) * matrix_size + i] =
        w[(matrix_size - 1) * matrix_size + i] = 0;
  }

  for (i = matrix_size; i < (matrix_size * matrix_size) - matrix_size; i++) {
    if (i % matrix_size == 0 || (i - (matrix_size - 1)) % matrix_size == 0) {
      continue;
    }
    u[i] = 75;
  }

  int num_iter = 0;

  while (globalDiff > EPSILON && num_iter < MAX_ITER) {
    globalDiff = 0.0;

    auto calc_loop_start = std::chrono::high_resolution_clock::now();
    for (i = matrix_size + 1; i < (matrix_size * matrix_size) - matrix_size - 1;
         i++) {
      if (i % matrix_size == 0 || (i - (matrix_size - 1)) % matrix_size == 0) {
        continue;
      }

      w[i] =
          (u[i - 1] + u[i + 1] + u[i - matrix_size] + u[i + matrix_size]) / 4;
      auto difference = fabs(w[i] - u[i]);
      if (difference > globalDiff) {
        globalDiff = difference;
      }
    }

    num_iter++;
    tmp = u;
    u = w;
    w = tmp;
  }

  auto full_calc_finish = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = full_calc_finish - full_calc_start;
  //   for (i = 0; i < matrix_size; i++) {
  //     for (j = 0; j < matrix_size; j++) {
  //       printf("%6.2f ", u[i * matrix_size + j]);
  //     }
  //     putchar('\n');
  //   }

  delete[] u;
  delete[] w;
  auto time = elapsed.count();
  printf("Finished in %d iterations. Elapsed time: %f\n", num_iter, time);
  return time;
}

int main(int argc, char *argv[]) {
  int m_size = 1000;
  int n_runs = 1;
  char *file;
  if (argc < 4) {
    cout << "please specify the matrix size, num runs and the stats output "
            "file path.";
  } else {
    m_size = atoi(argv[1]);
    n_runs = atoi(argv[2]);
    file = argv[3];
  }

  double tot_time = 0;
  for (int i = 1; i <= n_runs; i++) {
    tot_time += run_matrix_as_array(m_size);
  }

  double avg = tot_time / n_runs;

  if (!file) {
    printf("Average time = %f\n", avg);
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
  outputFile << m_size << "," << avg << endl;
  outputFile.close();
  exit(0);
}
