/**
 * matrix.h
 * Defines the matrix class.
 * ------------------------------------------------------
 * Copyright (c) 2020 Endi Zhupani
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction, 
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
 * is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or 
 * substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "common.h"
#include "base_matrix.h"
#include "mpi.h"
#include "math.h"
#ifndef MATRIX_H
#define MATRIX_H
namespace pde_solver::data::cpu_distr
{
     /**
 * @brief Class that defines a distributed matrix that is partitioned in blocks and distributed with MPI to multiple processors
 * 
 */
     class Matrix : public common::BaseMatrix
     {
     private:
          // points that are used by the stencil calcuation of the points in the left border of the partition
          double *left_ghost_points_;

          // points taht are used by the stencil calcuation of the points in the right border of the partition
          double *right_ghost_points_;

          // number of rows or columns that will serve as the halo of the partition
          int halo_size_;

          // width of the current partition
          int partition_width_;

          // height of the current partition
          int partition_height_;

          // number of the process that is working with the current partition
          int proc_id_;

          // number of processors being used by this matrix
          int proc_count_;

          // Number of partitions along the y-dimension
          int y_partitions_;

          // Number of partitions along the x-dimension
          int x_partitions_;

          // Neighbours on the top, right, bottom and left of the partition
          PartitionNeighbour neighbours[4];

          // Coordnates of the partition in the cartesian topology
          int partition_coords_[2];

          // Indicates whether the distributed matrix has been initialized or not.
          bool is_initialized_;

          // Iteration when the global reduction was performed for the last time.
          int last_global_reduction_iteration_;

          // Communicator object that uses a cartesian graph topology
          MPI_Comm cartesian_communicator_;

          // The max difference calculated after a Jacobian sweep operation
          double current_max_difference_;

          // Number of processes in each dimension. The first element determines the number processes along the vertical dimension the second, the number of processes along the horizontal dimension.
          int processes_per_dimension_[2];

          int right_grid_border_coord;
          int left_grid_border_coord;
          int bottom_grid_border_coord;
          int top_grid_border_coord;

          // Pointer to the final row of inner_data
          double *bottom_ghost;

          // Pointer to the first row of the inner_data. Technically inner_data can be used with the same result but this was supplied for completeness.
          double *top_ghost;

          /**
     * @brief Initializes the MPI context
     * 
     * @param argc Number of arguments provided by the user
     * @param argv Arguments provided by the user
     */
          void InitializeMPI(int argc, char *argv[]);

          /**
       * @brief Gets the rank of the process by the coordinates in the cartesian grid.
       * 
       * @param coords 
       * @param rank 
       */
          void RankByCoordinates(const int coords[2], int *rank);

          /**
       * @brief Initializes the left border of the partition and left ghost points that are part of the halo. 
       * 
       * @param inner_value Inner value of the global matrix. Needed because on inner partitions the left border of the partition will be the inner value.
       * @param left_border_value 
       * @param right_border_value 
       * @param bottom_border_value 
       * @param top_border_value 
       */
          void InitLeftBorderAndGhost(double inner_value, double left_border_value, double bottom_border_value, double top_border_value);

          /**
 * @brief 
 * 
 * @param inner_value 
 * @param right_border_value 
 * @param bottom_border_value 
 * @param top_border_value 
 */
          void InitRightBorderAndGhost(double inner_value, double right_border_value, double bottom_border_value, double top_border_value);

     public:
          /**
     * @brief Construct a new Matrix object
     * 
     * @param halo_size size of the halo around partitions this is in number of columns or rows.
     * @param width width of the global matrix
     * @param height height of the global matrix
     */
          Matrix(int halo_size, int width, int height);

          /**
     * @brief Initializes the global matrix and a new MPI context
     * 
     * @param value Value to assing to all elements of the matrix
     * @param argc Number of arguments provided by the user
     * @param argv Arguments provided by the user
     */
          void Init(double value, int argc, char *argv[]);

          /**
     * @brief Initializes the global matrix and a new MPI context. This method assigns custom values to the borders of the matrix
     * 
     * @param inner_value Value to be assigned to non-bordering elements of the global matrix
     * @param left_border_value Value to be assigned to the left border of the global matrix
     * @param right_border_value Value to be assigned to the right border of the global matrix
     * @param bottom_border_value Value to be assigned to the bottom border of the global matrix
     * @param top_border_value Value to be assigned to the top border of the global matrix
     * @param argc Number of arguments provided by the user
     * @param argv Arguments provided by the user
     */
          void Init(double inner_value, double left_border_value, double right_border_value, double bottom_border_value, double top_border_value, int argc, char *argv[]);

          /**
     * @brief Sends all the borders of the partition to the corresponding neighbors and gets the halo values from these neighbors
     * 
     */
          void AllNeighbourExchange();

          void SetGlobal(double value, int row, int col);
          void SetLocal(double value, int row, int col);

          const double GetLocal(int partition_row, int partition_col);

          /**
     * @brief Sends the border of the partition to the specified neighbor and gets the halo values from that neighbor
     * 
     * @param exchange_target Target neighbor
     */
          void NeighbourExchange(PartitionNeighbour exchange_target);

          /**
     * @brief Performs a Jacobian sweep of the current partition
     * 
     * @return double Maximum difference between the new and the old values
     */
          double LocalSweep(Matrix newMatrix);

          /**
     * @brief Gets the global max difference by performing a reduction operation for all processors
     * 
     * @return double Global max difference
     */
          double GlobalDifference();

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
 * @brief Gets the full set of points from the global matrix. Note that this will gather all points from all processes in the cartesian grid.
 * 
 */
          void ShowMatrix();

          void Synchronize();

          /**
 * @brief Puts together the data in the partition
 * 
 * @return double* 
 */
          double *AssemblePartition();

          bool IsTopBorder();
          bool IsBottomBorder();
          bool IsLeftBorder();
          bool IsRightBorder();

          void Finalize();

          /**
 * @brief Get the Partition Coordinates for the processor with the specified rank
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

} // namespace pde_solver::data::cpu_distr
#endif // !MATRIX_H
