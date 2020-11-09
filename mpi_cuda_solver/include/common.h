/**
 * common.h 
 * Defines common data structures
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
#include "mpi.h"
#include <string>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#ifndef COMMON_H
#define COMMON_H

/**
 * \brief This macro checks return value of the CUDA runtime call and exits
 *        the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value)                                          \
    {                                                                     \
        cudaError_t _m_cudaStat = value;                                  \
        if (_m_cudaStat != cudaSuccess)                                   \
        {                                                                 \
            fprintf(stderr, "Error %s at line %d in file %s\n",           \
                    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

/**
 * @brief The type of the partition neighbour
 * 
 */
enum PartitionNeighbourType
{
    BOTTOM_NEIGHBOUR,
    TOP_NEIGHBOUR
};

/**
 * @brief The neighbour of the partition. Identified by the processor number and the position relative to the target
 * 
 */
struct PartitionNeighbour
{
public:
    // The id of the processor that holds this neighbour
    int id;
    // Defines the type which determines the position of the neighbor
    enum PartitionNeighbourType type;

    std::string GetNeighborId()
    {
        if (id == MPI_PROC_NULL)
        {
            return "NULL";
        }

        return std::to_string(id);
    }
};

struct MatrixConfiguration
{
public:
    int gpu_number;
    int n_rows;
    int n_cols;
    float cpu_perc;
};

enum StorageType
{
    ROW_MAJOR,
    COL_MAJOR
};

struct GPUStream
{
public:
    // The device number on which the stream is allocated.
    int gpu_id;

    // The stream created on the gpu
    cudaStream_t stream;

    float *d_data;
    float *h_data;

    int halo_points_host_start;
    // Region index where the part of the data that needs to be calculated by the GPU starts
    // Important to note here is:
    // 1. For the top and bottom borders, this indicates the column index
    // 2. For the left and right borders as well as the inner region this indicates the row number.
    // 3. the acutal data transferred to the GPU will contain two more rows/columns on each end that represent the halo.
    // 4. This value is to be offset to get the actual starting point.
    //  For the left and right border, it should be offset by 1 becuase the first point of the border is calculated by the CPU.
    //  For the top and bottom border it should be offset by 2 because the first point of the row belongs to the side borders and the first point of the border is calculated by the CPU.
    //  For the inner points is should be offset by two because the first and second row if the inner data are the top halo and the top border.
    int gpu_region_start;
    int gpu_data_width;
    int gpu_data_height;
    bool is_contiguous_on_host;
};

struct ExecutionStats
{
public:
    double total_border_calc_time;
    double total_inner_points_time;
    double total_idle_comm_time;
    double total_sweep_time;
    double total_time_reducing_difference;
    int n_diff_reducions;
    int n_sweeps;

    void print_to_console()
    {
        if (n_sweeps == 0)
        {
            n_sweeps = 1;
        }

        if (n_diff_reducions == 0)
        {
            n_diff_reducions = 1;
        }
        printf("Total time spent executing Jacobi Iterations: %f\n\
        Total time executing border point calculations: %f\n\
        Total time executing inner point calculations: %f\n\
        Total time waiting for communication to finish: %f\n\
        Total time reducing and exchanging the global difference: %f\n\
        Avg time spent executing Jacobi Iterations: %f\n\
        Avg time executing border point calculations: %f\n\
        Avg time executing inner point calculations: %f\n\
        Avg time waiting for communication to finish: %f\n\
        Avg time reducing and exchanging the global difference: %f\n\
        Number of iterations: %d\n\
        Number of difference reductions: %d\n",
               total_sweep_time, total_border_calc_time, total_inner_points_time, total_idle_comm_time, total_time_reducing_difference, (total_sweep_time / n_sweeps), (total_border_calc_time / n_sweeps), (total_inner_points_time / n_sweeps), (total_idle_comm_time / n_sweeps), (total_time_reducing_difference / n_diff_reducions), n_sweeps, n_diff_reducions);
    }

    void print_to_file(char *file_path)
    {
    }
};

#endif // !COMMON_H
