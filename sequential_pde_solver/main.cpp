#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>
#define EPSILON 0.01
#define MAX_ITER 50

double run_matrix_as_array(int matrix_size) {
  double start = clock();
  std::cout << "Matrix size:" << matrix_size << " rows by " << matrix_size
            << " columns" << std::endl;
  int i, j;
  double globalDiff = 500;
  float *u = new float[matrix_size * matrix_size];
  float *w = new float[matrix_size * matrix_size];
  float *tmp;
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
    for (i = matrix_size + 1; i < (matrix_size * matrix_size) - matrix_size - 1;
         i++) {
      if (i % matrix_size == 0 || (i - (matrix_size - 1)) % matrix_size == 0) {
        continue;
      }

      w[i] =
          (u[i - 1] + u[i + 1] + u[i - matrix_size] + u[i + matrix_size]) / 4;
      double difference = fabs(w[i] - u[i]);
      // if (difference < 0) {
      //   difference *= -1;
      // }
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
  //   for (i = 0; i < matrix_size; i++) {
  //     for (j = 0; j < matrix_size; j++) {
  //       printf("%6.2f ", u[i * matrix_size + j]);
  //     }
  //     putchar('\n');
  //   }

  delete[] u;
  delete[] w;

  std::cout << "Finished in " << num_iter
            << " iterations. Elapsed time: " << elapsed << std::endl;
  return elapsed;
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
    // printf("3\n");
    m_size = atoi(argv[1]);
    // printf("4\n");
    n_runs = atoi(argv[2]);
    // printf("5\n");
    file = argv[3];
    // printf("6\n");
  }
  // printf("7\n");
  double tot_time = 0;
  for (int i = 1; i <= n_runs; i++) {
    // printf("8\n");
    tot_time += run_matrix_as_array(m_size);
    // printf("9\n");
  }

  double avg = tot_time / n_runs;
  // printf("10\n");
  if (!file) {
    std::cout << "Average time =" << avg << std::endl;
    exit(0);
  }
  printf("11\n");
  std::ifstream f(file);
  bool is_empty = f.peek() == std::ifstream::traits_type::eof();
  f.close();
  printf("12\n");
  std::ofstream outputFile;
  outputFile.open(file, std::ios_base::app);
  if (is_empty) {
    outputFile << "id,avg_time\n";
  }
  outputFile << m_size << "," << avg << "\n";
  outputFile.close();
  exit(0);
}
