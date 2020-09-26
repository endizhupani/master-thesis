/**
 * matrix.cpp
 * Implementation for the classes defined in /include/matrix.h
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

#include "matrix.h"

Matrix::Matrix(int halo_size, int width, int height)
{
    this->halo_size_ = halo_size;
    this->matrix_width_ = width;
    this->matrix_height_ = height;
}

MPI_Comm Matrix::GetCartesianCommunicator()
{
    return this->cartesian_communicator_;
}

void Matrix::Init(double value, int argc, char *argv[])
{
    this->Init(value, value, value, value, value, argc, argv);
}

void Matrix::Init(double inner_value, double left_border_value, double right_border_value, double bottom_border_value, double top_border_value, int argc, char *argv[])
{
    if (!is_initialized_)
    {
        return;
    }
}

void Matrix::InitializeMPI(int argc, char *argv[])
{
    if (is_initialized_)
    {
        // TODO(Endi): Replace this with a custom exception
        throw new std::logic_error("The MPI context is already initialized");
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count_);
    int n_dim = 2;
    int periods[2] = {false, false};
    int dims[2];
    MPI_Dims_create(proc_count_, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, n_dim, dims, periods, false, &this->cartesian_communicator_);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id_);
}