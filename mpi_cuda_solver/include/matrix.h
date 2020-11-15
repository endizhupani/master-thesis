/**
 * matrix.h
 * Defines the matrix class.
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

#include "base_matrix.h"
#include "common.h"
#include "kernels.cuh"
#include "math.h"
#include "mpi.h"
#include <array>
#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>

#ifndef MATRIX_H
#define MATRIX_H
namespace pde_solver {
/**
* @brief Class that defines a distributed matrix that is partitioned in blocks
and distributed with MPI to multiple processors
*
*/
class Matrix : public BaseMatrix {
private:
#pragma region Generic matrix configuration

  MatrixConfiguration matrix_config_;

  // number of rows or columns that will serve as the halo of the partition
  int halo_size_;

  // height of the current partition. It does not include the top and bottom
  int partition_height_;

  // Processor number that is handling the calculations for this partition
  int proc_id_;

  // Total number of processors in use for the global calculation
  int proc_count_;

  // Number of partitions along the y-dimension
  int y_partitions_;

#pragma endregion

#pragma region Communication related

  // Coordnates of the partition in the cartesian topology
  // Remark: the first value is the 'y' coordinate
  int partition_coords_[2];

  // Number of processes in each dimension. The first element determines the
  // number processes along the vertical dimension the second, the number of
  // processes along the horizontal dimension.
  int processes_per_dimension_[2];

  // Neighbours on the top and bottom of the partition
  std::array<PartitionNeighbour, 2> neighbours;

  // Communicator object that uses a cartesian graph topology
  MPI_Comm cartesian_communicator_;

  // True if the MPI context has been initialized.
  bool is_initialized_;
#pragma endregion

#pragma region Matrix Data

  int right_grid_border_coord_;
  int left_grid_border_coord_;
  int bottom_grid_border_coord_;
  int top_grid_border_coord_;

  // Pointer to the final row of inner_data
  float *bottom_halo_;

  // Pointer to the first row of the inner_data. Technically inner_data can be
  // used with the same result but this was supplied for completeness.
  float *top_halo_;

  float initial_inner_value_;
  float initial_left_value_;
  float initial_right_value_;
  float initial_top_value_;
  float initial_bottom_value_;

#pragma endregion

#pragma region Calculation management

  // Iteration when the global reduction was performed for the last time.
  int last_global_reduction_iteration_;

  // The max difference calculated after a Jacobian sweep operation
  float current_max_difference_;

  // Wrappers for reduction operations that are performed on the inner data
  // In this particular case, these are used to calculate the differences
  // between the old and new matrices
  std::vector<GpuReductionOperation> inner_data_reduction_plans_;

  // Streams for the border calculations. 2 streams for the top and bottom
  // border.
  GpuExecution border_execution_plans_[2];

  // Streams that will be used do calculate the inner data of the partition.
  // There will be one stream per GPU. This needs to be a vector because the
  // number of GPUs is not known in advance.
  std::vector<GpuExecution> inner_data_execution_plans_;

#pragma endregion

  /**
   * @brief Initializes the MPI context
   *
   * @param argc Number of arguments provided by the user
   * @param argv Arguments provided by the user
   */
  void InitializeMPI(int argc, char *argv[]);

  /**
   * @brief Gets the rank of the process by the coordinates in the cartesian
   * grid.
   *
   * @param coords
   * @param rank
   */
  void RankByCoordinates(const int coords[2], int *rank);

  /**
   * @brief Initializes the partition data.
   *
   * @param inner_value inner value of the global matrix
   * @param left_border_value left border value of the global matrix
   * @param right_border_value right border value of the global matrix
   * @param bottom_border_value bottom border value of the global matrix
   * @param top_border_value top border value of the global matrix
   */
  void InitData(float inner_value, float left_border_value,
                float right_border_value, float bottom_border_value,
                float top_border_value);
  /**
   * @brief Initializes the left border of the partition and left ghost points
   * that are part of the halo.
   *
   * @param inner_value Inner value of the global matrix. Needed because on
   * inner partitions the left border of the partition will be the inner value.
   * @param left_border_value
   * @param right_border_value
   * @param bottom_border_value
   * @param top_border_value
   */
  void InitLeftBorderAndGhost(float inner_value, float left_border_value,
                              float bottom_border_value,
                              float top_border_value);

  /**
   * @brief
   *
   * @param inner_value
   * @param right_border_value
   * @param bottom_border_value
   * @param top_border_value
   */
  void InitRightBorderAndGhost(float inner_value, float right_border_value,
                               float bottom_border_value,
                               float top_border_value);

  /**
   * @brief Configures the GPU Execution. Creates the necessary streams for each
   * available gpu that will be used for calculations and memory copy operations
   */
  void ConfigGpuExecution();

  /**
   * @brief Allocates the memory on the GPU and CPU
   *
   */
  void AllocateMemory();

  /**
   * @brief Executes the GPU calculation by overlapping kernel executions and
   * data transfers when possible
   *
   * @param new_matrix matrix where the calculation will be stored.
   * @param gpu_execution_plan GPU Execution plan to be used for the current
   * execution.
   */
  void ExecuteGpuWithConcurrentCopy(Matrix new_matrix,
                                    GpuExecution gpu_execution_plan);

public:
  void Deallocate();

  /**
   * @brief Gets the gpu execution plan for the part of the inner data of the
   * matrix that is next to the part being calculated by the CPU
   *
   * @return const GpuExecution&
   */
  const GpuExecution &GetCpuAdjacentInnerDataGpuPlan();

  /**
   * @brief Gets the inner data calculation gpu execution plan for the specified
   * gpu id.
   *
   * @param gpu_id
   * @return const GpuExecution&
   */
  const GpuExecution &GetInnerDataPlanForGpuId(int gpu_id);

  /**
   * @brief Gets the reduction operation details for the specified gpu id
   *
   * @param gpu_id
   * @return const GpuReductionOperation&
   */
  const GpuReductionOperation &GetInnerReductionOperation(int gpu_id);
  const GpuExecution GetLeftBorderStream();
  const GpuExecution GetRightBorderStream();

  /**
   * @brief Construct a new Matrix object
   *
   * @param halo_size size of the halo around partitions this is in number of
   * columns or rows.
   * @param width width of the global matrix
   * @param height height of the global matrix
   */
  Matrix(int halo_size, int width, int height);

  /**
   * @brief Construct a new Matrix object
   *
   * @param config
   */
  Matrix(MatrixConfiguration config);

  /**
   * @brief Initializes the global matrix and a new MPI context
   *
   * @param value Value to assing to all elements of the matrix
   * @param argc Number of arguments provided by the user
   * @param argv Arguments provided by the user
   */
  void Init(float value, int argc, char *argv[]);

  /**
   * @brief Initializes the global matrix and a new MPI context. This method
   * assigns custom values to the borders of the matrix
   *
   * @param inner_value Value to be assigned to non-bordering elements of the
   * global matrix
   * @param left_border_value Value to be assigned to the left border of the
   * global matrix
   * @param right_border_value Value to be assigned to the right border of the
   * global matrix
   * @param bottom_border_value Value to be assigned to the bottom border of the
   * global matrix
   * @param top_border_value Value to be assigned to the top border of the
   * global matrix
   * @param argc Number of arguments provided by the user
   * @param argv Arguments provided by the user
   */
  void Init(float inner_value, float left_border_value,
            float right_border_value, float bottom_border_value,
            float top_border_value, int argc, char *argv[]);

  /**
   * @brief Sends all the borders of the partition to the corresponding
   * neighbors and gets the halo values from these neighbors
   *
   */
  void AllNeighbourExchange();

  void SetGlobal(float value, int row, int col);
  void SetLocal(float value, int row, int col);

  const float GetLocal(int partition_row, int partition_col);

  const PartitionNeighbour GetNeighbour(PartitionNeighbourType neighbour_type);

  /**
   * @brief Sends the border of the partition to the specified neighbor and gets
   * the halo values from that neighbor
   *
   * @param exchange_target Target neighbor
   */
  void NeighbourExchange(PartitionNeighbour exchange_target);

  /**
   * @brief Performs a Jacobian sweep of the current partition
   *
   * @return float Maximum difference between the new and the old values
   */
  float LocalSweep(Matrix newMatrix, ExecutionStats *execution_stats);

  /**
   * @brief Gets the global max difference by performing a reduction operation
   * for all processors
   *
   * @return float Global max difference
   */
  const float GlobalDifference(ExecutionStats *execution_stats);

  /**
   * @brief Prints information about the current partition.
   *
   */
  void PrintMatrixInfo();

  /**
   * @brief Prints the current partition data
   *
   */
  void PrintPartitionData();

  /**
   * @brief Prints all the partitions in this matrix
   *
   */
  void PrintAllPartitions();

  /**
   * @brief Gets the full set of points from the global matrix. Note that this
   * will gather all points from all processes in the cartesian grid.
   *
   */
  void ShowMatrix();

  void Synchronize();

  Matrix CloneShell();

  /**
   * @brief Puts together the data in the partition
   *
   * @return float*
   */
  float *AssemblePartition();

  /**
   * @brief Checks if the partition is a partition that contains the top border
   * of the matrix.
   *
   * @return true if the partition contains the top border of the matrix.
   * @return false if the partition does not contain the top border of the
   * matrix.
   */
  bool IsTopBorder();

  /**
   * @brief Checks if the partition is a partition that contains the bottom
   * border of the matrix.
   *
   * @return true if the partition contains the bottom border of the matrix.
   * @return false if the partition does not contain the bottom border of the
   * matrix.
   */
  bool IsBottomBorder();

  void Finalize();

  /**
   * @brief Get the Partition Coordinates for the processor with the specified
   * rank
   *
   * @param rank
   * @return int*
   */
  void GetPartitionCoordinates(int rank, int *partition_coords);

#pragma region Getters and Setters

  /**
   * @brief Get the Cartesian Communicator object
   *
   * @return MPI_Comm
   */
  MPI_Comm GetCartesianCommunicator();
#pragma endregion
};
} // namespace pde_solver

#endif // !MATRIX_H
