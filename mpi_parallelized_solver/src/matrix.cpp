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
        for (int i = 0; i < 2; i++)
        {
            this->processes_per_dimension_[i] = 0;
        }
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

        if (this->matrix_width_ % this->processes_per_dimension_[0] != 0)
        {
            throw new std::logic_error("The width of the matrix must be divisable by the number of processes along the x dimension: " + std::to_string(this->processes_per_dimension_[0]));
        }
        this->partition_width_ = this->matrix_width_ / this->processes_per_dimension_[0];

        if (this->matrix_height_ % this->processes_per_dimension_[1] != 0)
        {
            throw new std::logic_error("The height of the matrix must be divisable by the number of processes along the y dimension: " + std::to_string(this->processes_per_dimension_[0]));
        }

        this->partition_height_ = this->matrix_height_ / this->processes_per_dimension_[1];
        int inner_data_width = (this->partition_width_ - 2);
        this->inner_points_ = new double[(this->partition_height_ + 2) * inner_data_width]; // Since the left and right borders of the partitions are stored separately, there is no need to store them on teh inner points array.

        int grid_left_border_coord = 0;
        int grid_top_border_coord = 0;
        int grid_right_border_coord = this->processes_per_dimension_[0] - 1;
        int grid_bottom_border_coord = this->processes_per_dimension_[1] - 1;

        // assing the left border values. If the partition is on the border of the cartesian grid,
        // the left border values should receive the values from the parameter 'left_border_value'
        this->left_border_ = new double[this->partition_height_];
        this->left_ghost_points_ = new double[this->partition_height_];
        this->right_border_ = new double[this->partition_height_];
        this->right_ghost_points_ = new double[this->partition_height_];
        if (this->partition_coords_[1] == grid_left_border_coord)
        {
            for (int i = 0; i < this->partition_height_; i++)
            {
                left_border_[i] = left_border_value;
            }
        }
        else if (this->partition_coords_[1] < grid_right_border_coord) // inner partition
        {
            // if on the top row the first value of the left border will be the top border value.
            int start = 0;
            if (this->partition_coords_[0] == grid_top_border_coord)
            {
                this->left_border_[start] = top_border_value;
                this->left_ghost_points_[start] = top_border_value;
                start++;
            }

            // if on the borrom row the last value of the left border will be the bottom border value.
            int end = this->partition_height_ - 1;
            if (this->partition_coords_[0] == grid_bottom_border_coord)
            {
                this->left_border_[end] = bottom_border_value;
                this->left_ghost_points_[end] = bottom_border_value;
                end--;
            }

            for (int i = start; i <= end; i++)
            {
                this->left_border_[i] = inner_value;
                this->left_ghost_points_[i] = inner_value;
            }
        }

        //assign the right border values of the global matrix
        if (this->partition_coords_[1] == (this->processes_per_dimension_[0] - 1))
        {

            for (int i = 0; i < this->partition_height_; i++)
            {
                right_border_[i] = right_border_value;
            }
        }
        else if (this->partition_coords_[1] > 0)
        {

            int start = 0;
            if (this->partition_coords_[0] == 0)
            {
                this->right_border_[start] = top_border_value;
                this->right_ghost_points_[start] = top_border_value;
                start++;
            }

            // if on the borrom row the last value of the right border will be the bottom border value.
            int end = this->partition_height_ - 1;
            if (this->partition_coords_[0] == (this->processes_per_dimension_[1] - 1))
            {
                this->right_border_[end] = bottom_border_value;
                this->right_ghost_points_[end] = bottom_border_value;
                end--;
            }

            for (int i = start; i <= end; i++)
            {
                this->right_border_[i] = inner_value;
                this->right_ghost_points_[i] = inner_value;
            }
        }

        for (int i = 0; i < this->partition_height_; i++)
        {
            for (int j = 0; j < inner_data_width; j++)
            {
                if (this->partition_coords_[0] == grid_top_border_coord && i == 0)
                {
                    this->inner_points_[j] = top_border_value;
                }
                else if (this->partition_coords_[0] == grid_bottom_border_coord && i == (this->partition_height_ - 1))
                {
                    this->inner_points_[(i * inner_data_width) + j] = bottom_border_value;
                }
                else
                {
                    this->inner_points_[(i * inner_data_width) + j] = inner_value;
                }
            }
        }
    }

    void Matrix::PrintMatrixInfo()
    {
        if (this->proc_id_ == 0)
        {
            printf("Number of processes: %d\n", this->proc_count_);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: (" << this->partition_coords_[0] << ", " << this->partition_coords_[1] << "); Top ID: " << this->neighbours[0].GetNeighborId() << "; Right ID: " << this->neighbours[1].GetNeighborId() << "; Bottom ID: " << this->neighbours[2].GetNeighborId() << "; Left ID: " << this->neighbours[3].GetNeighborId() << std::endl;
    }

    void Matrix::RankByCoordinates(const int coords[2], int *rank)
    {
        if (coords[0] < 0 || coords[1] < 0 || coords[0] >= processes_per_dimension_[0] || coords[1] >= processes_per_dimension_[1])
        {
            *rank = MPI_PROC_NULL;
            return;
        }
        MPI_Cart_rank(this->GetCartesianCommunicator(), coords, rank);
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
        int periods[2] = {0, 0};
        //processes_per_dimension_[0] = 2;
        //processes_per_dimension_[1] = 2;
        MPI_Dims_create(proc_count_, 2, processes_per_dimension_);
        MPI_Cart_create(MPI_COMM_WORLD, n_dim, processes_per_dimension_, periods, 1, &this->cartesian_communicator_);
        MPI_Comm_rank(this->GetCartesianCommunicator(), &proc_id_);
        // Fetch the coordinates of the current partition
        MPI_Cart_coords(this->GetCartesianCommunicator(), this->proc_id_, 2, this->partition_coords_);

        this->neighbours[0].type = PartitionNeighbourType::TOP_NEIGHBOUR;
        int neighbor_coords[2] = {this->partition_coords_[0] - 1, this->partition_coords_[1]};
        this->RankByCoordinates(neighbor_coords, &this->neighbours[0].id);

        this->neighbours[1]
            .type = PartitionNeighbourType::RIGHT_NEIGHBOUR;
        neighbor_coords[0] = this->partition_coords_[0];
        neighbor_coords[1] = this->partition_coords_[1] + 1;
        this->RankByCoordinates(neighbor_coords, &this->neighbours[1].id);

        this->neighbours[2].type = PartitionNeighbourType::BOTTOM_NEIGHBOUR;
        neighbor_coords[0] = this->partition_coords_[0] + 1;
        neighbor_coords[1] = this->partition_coords_[1];
        this->RankByCoordinates(neighbor_coords, &this->neighbours[2].id);

        this->neighbours[3].type = PartitionNeighbourType::LEFT_NEIGHTBOUR;
        neighbor_coords[0] = this->partition_coords_[0];
        neighbor_coords[1] = this->partition_coords_[1] - 1;
        this->RankByCoordinates(neighbor_coords, &this->neighbours[3].id);
    }

    void Matrix::Finalize()
    {
        delete[] left_ghost_points_;
        delete[] right_ghost_points_;
        delete[] left_border_;
        delete[] right_border_;
        delete[] inner_points_;
        MPI_Finalize();
    }

    double *Matrix::AsembleMatrix()
    {

        throw "Not implemented";
        // TODO (endizhupani@uni-muenster.de): To implment this, first the left and right column of the partition need to be merged into one array and then an all gather operation should be performed only on the root process.
        //return this->inner_points_;
    }
} // namespace pde_solver::data::cpu_distr
