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

namespace pde_solver::data::cpu_distr
{
    Matrix::Matrix(int halo_size, int width, int height) : pde_solver::data::common::BaseMatrix(width, height)
    {
        this->halo_size_ = halo_size;
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
            this->InitializeMPI(argc, argv);
        }
    }

    void Matrix::InitializeMPI(int argc, char *argv[])
    {
        if (is_initialized_)
        {
            // TODO(endizhupani@uni-muenster.de): Replace this with a custom exception
            throw new std::logic_error("The MPI context is already initialized");
        }
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &proc_count_);
        int n_dim = 2;
        int periods[2] = {false, false};
        int dims[2];
        MPI_Dims_create(proc_count_, 2, dims);
        MPI_Cart_create(MPI_COMM_WORLD, n_dim, dims, periods, false, &this->cartesian_communicator_);
        MPI_Comm_rank(this->GetCartesianCommunicator(), &proc_id_);

        // Fetch the coordinates of the current partition
        MPI_Cart_coords(this->GetCartesianCommunicator(), this->proc_id_, 2, this->partition_coords);

        this->neighbours[0].type = PartitionNeighbourType::TOP_NEIGHBOUR;
        MPI_Cart_shift(this->GetCartesianCommunicator(), 1, 1, &proc_id_, &this->neighbours[0].id);

        this->neighbours[1].type = PartitionNeighbourType::RIGHT_NEIGHBOUR;
        MPI_Cart_shift(this->GetCartesianCommunicator(), 0, 1, &proc_id_, &this->neighbours[1].id);

        this->neighbours[2].type = PartitionNeighbourType::BOTTOM_NEIGHBOUR;
        MPI_Cart_shift(this->GetCartesianCommunicator(), 1, -1, &proc_id_, &this->neighbours[2].id);

        this->neighbours[3].type = PartitionNeighbourType::LEFT_NEIGHTBOUR;
        MPI_Cart_shift(this->GetCartesianCommunicator(), 0, -1, &proc_id_, &this->neighbours[3].id);
    }

    double *Matrix::GetAllPoints()
    {
        throw "Not implemented";
        // TODO (endizhupani@uni-muenster.de): To implment this, first the left and right column of the partition need to be merged into one array and then an all gather operation should be performed only on the root process.
        //return this->inner_points_;
    }
} // namespace pde_solver::data::cpu_distr
