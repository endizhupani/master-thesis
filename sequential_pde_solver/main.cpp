#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>
#define EPSILON 0.01
#define MAX_ITER 5000

double run_matrix_as_array(int matrix_size) {

  std::cout << "Matrix size:" << matrix_size << " rows by " << matrix_size
            << " columns" << std::endl;
  int i, j;
  double globalDiff = 500;
  float *u = new float[matrix_size * matrix_size];
  float *w = new float[matrix_size * matrix_size];
  float *tmp;
  for (i = 0; i < (matrix_size * matrix_size); i++) {
    u[i] = 75;
  }

  int num_iter = 0;
  double start = clock();
  while (globalDiff > EPSILON && num_iter < MAX_ITER) {
    globalDiff = 0.0;
    for (i = 0; i < (matrix_size * matrix_size); i++) {
      int row, col;
      row = i / matrix_size;
      col = i % matrix_size;
      float top, bottom, right, left;
      if (row == 0) {
        top = 100;
      } else {
        top = u[i - matrix_size];
      }
      if (row == matrix_size - 1) {
        bottom = 0;
      } else {
        bottom = u[i + matrix_size];
      }

      if (col == 0) {
        left = 100;
      } else {
        left = u[i - 1];
      }

      if (col == matrix_size - 1) {
        right = 100;
      } else {
        right = u[i + 1];
      }

      w[i] = (top + bottom + right + left) / 4;
      double difference = fabs(w[i] - u[i]);
      if (difference > globalDiff) {
        globalDiff = difference;
      }
    }

    num_iter++;
    tmp = u;
    u = w;
    w = tmp;
  }

  clock_t end = clock();
  double elapsed = double(end - start) / CLOCKS_PER_SEC;
  // for (i = 0; i < matrix_size; i++) {
  //   for (j = 0; j < matrix_size; j++) {
  //     printf("%6.2f ", u[i * matrix_size + j]);
  //   }
  //   putchar('\n');
  // }

  delete[] u;
  delete[] w;

  std::cout << "Finished in " << num_iter
            << " iterations. Elapsed time: " << elapsed << std::endl;
  return elapsed;
}

int main(int argc, char *argv[]) {
  int m_size = 512;
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
    tot_time += run_matrix_as_array(m_size);
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
