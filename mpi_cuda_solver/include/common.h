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
#ifndef COMMON_H
#define COMMON_H

#ifdef __CUDACC__
/**
 * \brief Macro for function type qualifiers __host__ __device__.
 *
 * Macro for function type qualifiers __host__ __device__. This macro is only
 * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
 * compiler will complain about function type qualifiers.
 */
#define MSL_USERFUNC __host__ __device__
/**
 * \brief Macro for function type qualifier __device__.
 *
 * Macro for function type qualifier __device__. This macro is only
 * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
 * compiler will complain about function type qualifiers.
 */
#define MSL_GPUFUNC __device__
/**
 * \brief Macro for function type qualifier __host__.
 *
 * Macro for function type qualifier __host__. This macro is only
 * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
 * compiler will complain about function type qualifiers.
 */
#define MSL_CPUFUNC __host__
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
#endif

/**
 * @brief The type of the partition neighbour
 * 
 */
enum PartitionNeighbourType
{
    LEFT_NEIGHTBOUR,
    RIGHT_NEIGHBOUR,
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
    int left_border;
    int right_border;
    int top_border;
    int bottom_border;
    int inner_value;

    int gpu_number;
    int n_rows;
    int n_cols;
    int requested_num_threads;
    int granted_num_threads;
};

#endif // !COMMON_H
