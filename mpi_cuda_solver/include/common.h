/**
 * common.h
 * Defines common data structures
 * ------------------------------------------------------
 * Copyright (c) 2020 Endi Zhupani
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
//#include "cuda.h"
#include "math.h"
#include "mpi.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

#ifndef COMMON_H
#define COMMON_H

/**
 * \brief This macro checks return value of the CUDA runtime call and exits
 *        the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void InitMPIContext(int argc, char *argv[]);
void FinalizeMPIContext();
int GetProcessId();
int End(int error_code);

/**
 * @brief The type of the partition neighbour
 *
 */
enum PartitionNeighbourType { BOTTOM_NEIGHBOUR, TOP_NEIGHBOUR };

/**
 * @brief The neighbour of the partition. Identified by the processor number and
 * the position relative to the target
 *
 */
struct PartitionNeighbour {
public:
  // The id of the processor that holds this neighbour
  int id;
  // Defines the type which determines the position of the neighbor
  enum PartitionNeighbourType type;

  std::string GetNeighborId() {
    if (id == MPI_PROC_NULL) {
      return "NULL";
    }

    return std::to_string(id);
  }
};

struct MatrixConfiguration {
public:
  int gpu_number;
  int n_rows;
  int n_cols;
  double cpu_perc;

  /**
   * @brief Determines the number of rows our the total that should be processed
   * on the GPU
   *
   * @param total_rows Total number of rows involved in the calculation
   * @return int Number of rows to be processed by the GPU
   */
  int GpuRows(int total_rows) { return ceil(total_rows * (1 - cpu_perc)); }

  std::string GetConfId() {
    return "s" + std::to_string(n_rows) + "_g" + std::to_string(gpu_number) +
           "_cp" + std::to_string(cpu_perc * 100);
  }
};

/**
 * @brief Structure describing a GPU Reduction operation
 *
 */
struct GpuReductionOperation {
public:
  // Id of the GPU where the reduction operation will be run on.
  int gpu_id;

  // pointer to the device vector that needs to be reduced
  float *d_vector_to_reduce;

  // length of the vector to reduce
  int vector_to_reduce_length;

  // pointer to the reduction result stored in the device's global memory
  float *d_reduction_result;

  float *h_reduction_result;

  // size in bytes for the temp data vector required by the reduction operation
  size_t tmp_data_size_in_bytes = 0;

  // temp data vector that is used by the reduction operation
  void *d_tmp_data = NULL;
};

struct GpuExecution {
#define DEFAULT_BLOCK_DIM 16
private:
  int gpu_block_size_x = 0;
  int gpu_block_size_y = 0;
  int gpu_grid_size_x = 0;
  int gpu_grid_size_y = 0;
  int gpu_data_width = 0;
  int gpu_data_height = 0;
  int gpu_region_start;
  bool concurrentKernelCopy;

  /**
   * @brief Tries to set the block sizes such that the blocks have the best
   * possible occupacy.
   *
   */
  void AdjustGridAndBlockSizes() {
    if (gpu_data_height % DEFAULT_BLOCK_DIM >=
        ((double)DEFAULT_BLOCK_DIM / 2)) {
      gpu_block_size_y = DEFAULT_BLOCK_DIM / 2;
    } else {
      gpu_block_size_y = DEFAULT_BLOCK_DIM;
    }

    gpu_grid_size_y = ceil((double)gpu_data_height / GetGpuBlockSizeY());

    if (gpu_data_width % DEFAULT_BLOCK_DIM >= ((double)DEFAULT_BLOCK_DIM / 2)) {
      gpu_block_size_x = DEFAULT_BLOCK_DIM / 2;
    } else {
      gpu_block_size_x = DEFAULT_BLOCK_DIM;
    }

    gpu_grid_size_x = ceil((double)gpu_data_width / GetGpuBlockSizeX());
  }

public:
  // The device number on which the stream is allocated.
  int gpu_id;

  // The stream created on the gpu
  cudaStream_t stream;
  cudaStream_t auxilary_copy_stream;
  cudaStream_t auxilary_calculation_stream;

  float *d_data;

  // A pointer to the first element from the host data that is also copied to
  // the device.
  float *h_d_data_mirror;

  int halo_points_host_start;

  bool is_contiguous_on_host;

  void Print() {
    printf("gpu_id: %d\ngpu_block_size_x: %d\ngpu_block_size_y: "
           "%d\ngpu_grid_size_x: %d\ngpu_grid_size_y: %d\ngpu_data_width: "
           "%d\ngpu_data_height: %d\ngpu_region_start: %d\n",
           gpu_id, gpu_block_size_x, gpu_block_size_y, gpu_grid_size_x,
           gpu_grid_size_y, gpu_data_width, gpu_data_height, gpu_region_start);
  }

  /**
   * @brief Sets the row of the full partition data object (including the top
   * and bottom halos of the partition data) where the GPU calculation starts
   *
   * @param value
   */
  void SetGpuCalculationStartRow(int value) { gpu_region_start = value; }

  /**
   * @brief Gets the first row of the region that is calculated by the GPU.
   * Skips any halo rows that might be in the GPU data. The row index is on the
   * full partition data with 0 being the first halo row
   *
   * @return const int
   */
  const int GetAbsoluteGpuCalculationStartRow() { return gpu_region_start; }

  /**
   * @brief Gets the first row of the region that is calculated by the GPU.
   * Skips any halo rows that might be in the GPU data. The row index is on the
   * partition data with 0 being the first row of the partition. The halo rows
   * are skipped
   *
   * @return const int
   */
  const int GetRelativeGpuCalculationStartRow() { return gpu_region_start - 1; }

  /**
   * @brief returns the value of the first row that should be copied to the GPU.
   * This includes any halo rows that will not be calculated, but are needed for
   * the calculation of other points in the matrix. The row index is on the
   * full partition data with 0 being the first halo row
   *
   * @return const int
   */
  const int GetGpuDataStartRow() { return gpu_region_start - 1; }

  void SetDeviceProperties(cudaDeviceProp deviceProp) {
    concurrentKernelCopy = deviceProp.deviceOverlap;
  }
  const bool GetConcurrentKernelAndCopy() { return concurrentKernelCopy; }
  /**
   * @brief Set the Gpu Data Height. Note that this should be the height of only
   * the data that is calculated by the GPU. it should not include halo cells.
   *
   * @param height
   */
  void SetGpuDataHeight(const int height) {
    gpu_data_height = height;
    if (gpu_data_width <= 0) {
      return;
    }

    AdjustGridAndBlockSizes();
  }

  /**
   * @brief Set the Gpu Data Width. Note that this should be the width of only
   * the data that is calculated by the GPU. it should not include halo cells.
   *
   * @param width
   */
  void SetGpuDataWidth(const int width) {
    gpu_data_width = width;
    if (gpu_data_height <= 0) {
      return;
    }

    AdjustGridAndBlockSizes();
  }

  /**
   * @brief Returns the height of the region that is calculated by the GPU. This
   * is the height of the data stored in the GPU minus the top and bottom GPU
   * halos
   *
   * @return int
   */
  int GetGpuCalculatedRegionHeight() { return gpu_data_height; }

  /**
   * @brief Returns the full height of the data section residing on the GPU
   *
   * @return int
   */
  int GetGpuDataHeight() { return gpu_data_height + 2; }

  /**
   * @brief Returns the width of the region that is calculated by the GPU. This
   * is the width of the data stored in the GPU minus the left and right columns
   * which are the left and right borders that are not calculated by the GPU
   *
   * @return int
   */
  int GetGpuCalculatedRegionWidth() { return gpu_data_width; }

  /**
   * @brief Returnds the full width of the data stored in the GPU.
   *
   * @return int
   */
  int GetGpuDataWidth() { return gpu_data_width + 2; }

  int GetGpuBlockSizeY() { return gpu_block_size_y; }

  int GetGpuBlockSizeX() { return gpu_block_size_x; }

  int GetGpuGridSizeX() { return gpu_grid_size_x; }

  int GetGpuGridSizeY() { return gpu_grid_size_y; }
};

struct ExecutionStats {
public:
  std::string id;
  double total_border_calc_time;
  double total_inner_points_time;
  double total_idle_comm_time;
  double total_sweep_time;
  double total_time_reducing_difference;
  double total_time_waiting_to_host_transfer;
  double total_time_waiting_to_device_transfer;
  double last_global_difference;
  double total_jacobi_time;
  int n_diff_reducions;
  int n_sweeps;

  void print_to_console() {
    if (n_sweeps == 0) {
      n_sweeps = 1;
    }

    if (n_diff_reducions == 0) {
      n_diff_reducions = 1;
    }
    printf("Total time spent executing Jacobi Iterations: %f\n\
        Total time executing border point calculations: %f\n\
        Total time executing inner point calculations: %f\n\
        Total time waiting for communication to finish: %f\n\
        Total time reducing and exchanging the global difference: %f\n\
        Total time waiting for data to be transferred to host: %f\n\
        Total time waiting for data to be transferred to device: %f\n\
        Avg time spent executing Jacobi Iterations: %f\n\
        Avg time executing border point calculations: %f\n\
        Avg time executing inner point calculations: %f\n\
        Avg time waiting for communication to finish: %f\n\
        Avg time reducing and exchanging the global difference: %f\n\
        Avg time waiting for data to be transferred to host: %f\n\
        Avg time waiting for data to be transferred to device: %f\n\
        Number of iterations: %d\n\
        Number of difference reductions: %d\n\
        Last difference: %f\n",
           total_sweep_time, total_border_calc_time, total_inner_points_time,
           total_idle_comm_time, total_time_reducing_difference,
           total_time_waiting_to_host_transfer,
           total_time_waiting_to_device_transfer, (total_sweep_time / n_sweeps),
           (total_border_calc_time / n_sweeps),
           (total_inner_points_time / n_sweeps),
           (total_idle_comm_time / n_sweeps),
           (total_time_reducing_difference / n_diff_reducions),
           (total_time_waiting_to_host_transfer / n_sweeps),
           (total_time_waiting_to_device_transfer / n_sweeps), n_sweeps,
           n_diff_reducions, last_global_difference);
  }

  void PrintHeaderToFile(char *file_path) {
    std::ifstream f(file_path);
    bool is_empty = f.peek() == std::ifstream::traits_type::eof();
    f.close();
    if (!is_empty) {
      return;
    }
    std::ofstream outputFile;
    outputFile.open(file_path, std::ios_base::app);
    outputFile
        << "id,total_border_calc_time,total_inner_points_time,total_idle_comm_"
           "time,total_sweep_time,total_time_reducing_difference,total_time_"
           "waiting_to_host_transfer,total_time_waiting_to_device_transfer,"
           "total_jacobi_time,"
           "last_global_difference,n_diff_reducions,n_sweeps\n";
    outputFile.close();
  }

  void PrintToFile(char *file_path) {
    std::ofstream outputFile;
    outputFile.open(file_path, std::ios_base::app);
    outputFile << id << "," << total_border_calc_time << ","
               << total_inner_points_time << "," << total_idle_comm_time << ","
               << total_sweep_time << "," << total_time_reducing_difference
               << "," << total_time_waiting_to_host_transfer << ","
               << total_time_waiting_to_device_transfer << ","
               << total_jacobi_time << "," << last_global_difference << ","
               << n_diff_reducions << "," << n_sweeps << "\n";
    outputFile.close();
  }
};

#endif // !COMMON_H
