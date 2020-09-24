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
#ifndef MATRIX_H
#define MATRIX_H

/**
 * @brief Class that defines a distributed matrix that is partitioned in blocks and distributed with MPI to multiple processors
 * 
 */
class Matrix
{
private:
    // array that holds the left border of the partition
    double *partition_left_border_;

    // array that holds the right border of the partition
    double *partition_right_border_;

    // points that are used by the stencil calcuation of the points in the left border of the partition
    double *left_ghost_points_;

    // points taht are used by the stencil calcuation of the points in the right border of the partition
    double *right_ghost_points_;

    // number of rows or columns that will serve as the halo of the partition
    int halo_size_;

    // global matrix width
    int matrix_width_;

    // global matrix height
    int matrix_height_;

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

public:
    Matrix(int halo_size, int width, int height, int processor_count);
    void Init(double value);
    void Init(double inner_value, double left_border_value, double right_border_value, double bottom_border_value, double top_border_value);
    void AllNeighbourExchange();
    void NeighbourExchange(PartitionNeighbour exchange_target);
    double LocalSweep();       // should return max difference for the local partition
    double GlobalDifference(); // should do an MPI all reduce to get the global difference from all partitions
};
#endif // !MATRIX_H
