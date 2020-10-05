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
        this->is_initialized_ = false;
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

        if (this->matrix_width_ % this->processes_per_dimension_[1] != 0)
        {

            throw new std::logic_error("The width of the matrix must be divisable by the number of processes along the x dimension: " + std::to_string(this->processes_per_dimension_[0]));
        }

        // This is the partition width without ghost points
        this->partition_width_ = this->matrix_width_ / this->processes_per_dimension_[1];

        if (this->matrix_height_ % this->processes_per_dimension_[0] != 0)
        {
            throw new std::logic_error("The height of the matrix must be divisable by the number of processes along the y dimension: " + std::to_string(this->processes_per_dimension_[0]));
        }

        // This is the partition height without the ghost points.
        this->partition_height_ = this->matrix_height_ / this->processes_per_dimension_[0];
        int inner_data_width = this->partition_width_;                          //(this->partition_width_ - 2);
        int inner_data_height = this->partition_height_ + 2;                    // Two extra rows to hold the top and bottom halo
        this->inner_points_ = new double[inner_data_height * inner_data_width]; // Since the left and right borders of the partitions are stored separately, there is no need to store them on teh inner points array.

        this->InitLeftBorderAndGhost(inner_value, left_border_value, bottom_border_value, top_border_value);
        this->InitRightBorderAndGhost(inner_value, right_border_value, bottom_border_value, top_border_value);

        // init top row. This will be the top ghost points which come from the bottom border of the partition above it.

        int row = 0;
        int col;

        // Top halo initialization
        // If the partition is on top border of the grid, there is no top halo to initialize.
        this->top_ghost = &this->inner_points_[0];
        if (!this->IsTopBorder())
        {
            if (this->IsLeftBorder())
            {
                // the value on the first column of the halo will come from the left border
                this->inner_points_[0] = left_border_value;
            }
            else
            {
                this->inner_points_[0] = inner_value;
            }

            if (this->IsRightBorder())
            {
                // the value on the last column of the halo will come from the right border
                this->inner_points_[inner_data_width - 1] = right_border_value;
            }
            else
            {
                this->inner_points_[inner_data_width - 1] = inner_value;
            }

            for (col = 1; col < inner_data_width - 1; col++)
            {
                this->inner_points_[col] = inner_value;
            }
        }

        // Bottom Halo initialization
        // if the partition is on the border level of the
        this->bottom_ghost = &this->inner_points_[(inner_data_height - 1) * inner_data_width];
        if (!this->IsBottomBorder())
        {
            row = inner_data_height - 1;
            if (this->IsLeftBorder())
            {
                // the value on the first column of the halo will come from the left border
                this->inner_points_[row * inner_data_width] = left_border_value;
            }
            else
            {
                this->inner_points_[row * inner_data_width] = inner_value;
            }

            if (this->IsRightBorder())
            {
                // the value on the last column of the halo will come from the right border
                this->inner_points_[inner_data_width * inner_data_height - 1] = right_border_value;
            }
            else
            {
                this->inner_points_[inner_data_width * inner_data_height - 1] = inner_value;
            }

            for (col = 1; col < inner_data_width - 1; col++)
            {
                this->inner_points_[row * inner_data_width + col] = inner_value;
            }
        }

        // Inner data
        for (int i = 1; i < inner_data_height - 1; i++)
        {
            int offset = i * inner_data_width;
            for (int j = 1; j < inner_data_width - 1; j++)
            {
                if (this->IsTopBorder() && i == 1)
                {
                    this->inner_points_[offset + j] = top_border_value;
                }
                else if (this->IsBottomBorder() && i == (inner_data_height - 2))
                {
                    this->inner_points_[offset + j] = bottom_border_value;
                }
                else
                {
                    this->inner_points_[offset + j] = inner_value;
                }
            }
        }

        MPI_Barrier(this->GetCartesianCommunicator());
    }

    void Matrix::InitLeftBorderAndGhost(double inner_value, double left_border_value, double bottom_border_value, double top_border_value)
    {

        // assing the left border values. If the partition is on the border of the cartesian grid,
        // the left border values should receive the values from the parameter 'left_border_value'
        this->left_border_ = new double[this->partition_height_];
        this->left_ghost_points_ = new double[this->partition_height_];

        if (this->IsLeftBorder())
        {
            for (int i = 0; i < this->partition_height_; i++)
            {
                left_border_[i] = left_border_value;
            }

            return;
        }

        int start = 0;
        if (this->IsTopBorder())
        {
            this->left_border_[start] = top_border_value;
            this->left_ghost_points_[start] = top_border_value;
            start++;
        }

        int end = this->partition_height_ - 1;
        if (this->IsBottomBorder())
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

    void Matrix::InitRightBorderAndGhost(double inner_value, double right_border_value, double bottom_border_value, double top_border_value)
    {
        this->right_border_ = new double[this->partition_height_];
        this->right_ghost_points_ = new double[this->partition_height_];

        //assign the right border values of the global matrix
        if (this->IsRightBorder())
        {
            for (int i = 0; i < this->partition_height_; i++)
            {
                right_border_[i] = right_border_value;
            }

            return;
        }
        int start = 0;
        if (this->IsTopBorder())
        {
            this->right_border_[start] = top_border_value;
            this->right_ghost_points_[start] = top_border_value;
            start++;
        }

        int end = this->partition_height_ - 1;
        if (this->IsBottomBorder())
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

    void Matrix::PrintMatrixInfo()
    {
        if (this->proc_id_ == 0)
        {
            printf("Number of processes: %d\n", this->proc_count_);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: (" << this->partition_coords_[0] << ", " << this->partition_coords_[1] << "); Top ID: " << this->neighbours[0].GetNeighborId() << "; Right ID: " << this->neighbours[1].GetNeighborId() << "; Bottom ID: " << this->neighbours[2].GetNeighborId() << "; Left ID: " << this->neighbours[3].GetNeighborId() << "Partition size: " << this->partition_width_ << "x" << this->partition_height_ << std::endl;
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

        MPI_Dims_create(this->proc_count_, n_dim, this->processes_per_dimension_);
        this->right_grid_border_coord = this->processes_per_dimension_[1] - 1;
        this->bottom_grid_border_coord = this->processes_per_dimension_[0] - 1;
        this->left_grid_border_coord = 0;
        this->top_grid_border_coord = 0;

        MPI_Cart_create(MPI_COMM_WORLD, n_dim, this->processes_per_dimension_, periods, 1, &this->cartesian_communicator_);
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

    double Matrix::LocalSweep()
    {
        throw "Not implemented";
    }

    double *Matrix::AssembleMatrix()
    {
        MPI_Barrier(this->GetCartesianCommunicator());
        // Holds the data returned by each processor. Processor with rank k places the data in row k-1 of the matrix.
        double **proc_buffers = new double *[this->proc_count_];

        proc_buffers[this->proc_id_] = this->AssemblePartition();
        double *matrix;
        int req_count = (this->proc_count_ - 1) * 2; // 1 send request for each of the n-1 non-root processors and n-1 receives on the root processor.
        MPI_Request requests[req_count];
        for (int i = 0; i < req_count; i++)
        {

            requests[i] = MPI_REQUEST_NULL;
        }

        // TODO (endizhupani@uni-muenster.de): Fetch everything from other partitions

        // Data have to be retrieved form each separate processor.
        if (this->proc_id_ != 0)
        {
            MPI_Isend(proc_buffers[this->proc_id_], this->partition_width_ * this->partition_height_, MPI_DOUBLE, 0, 0, this->GetCartesianCommunicator(), &requests[this->proc_id_ - 1]);
        }

        // if (this->proc_id_ != 0)
        // {
        //     delete[] proc_buffers[this->proc_id_];
        // }

        if (this->proc_id_ == 0)
        {
            for (int proc = 1; proc < this->proc_count_; proc++)
            {
                if (!(proc == this->proc_id_))
                {
                    continue;
                }

                auto el_count = this->partition_width_ * this->partition_height_;

                proc_buffers[proc] = new double[el_count];

                MPI_Irecv(proc_buffers[proc], el_count, MPI_DOUBLE, 0, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[req_count - this->proc_id_]);
            }
        }

        MPI_Status statuses[this->proc_count_ - 1];
        MPI_Waitall(this->proc_count_ - 1, requests, statuses);
        MPI_Barrier(this->GetCartesianCommunicator());
        printf("GOT HERE\n\n");
        if (this->proc_id_ == 0)
        {
            matrix = new double[this->matrix_height_ * this->matrix_width_];

            for (int i = 0; i < this->proc_count_; i++)
            {

                int coords[2];
                this->GetPartitionCoordinates(i, coords);

                for (int row = 0; row < this->partition_height_; row++)
                {

                    int matrix_row = coords[0] * this->partition_height_ + row;

                    int start_in_row = coords[1] * partition_width_;
                    printf("GOT HERE: (%d, %d)\n\n", i, row);
                    fflush(stdout);
                    std::copy(proc_buffers[i] + row * this->partition_width_, proc_buffers[i] + row * this->partition_width_ + this->partition_width_, matrix + (matrix_row * this->matrix_width_) + start_in_row);
                    /* code */
                }

                //delete[] proc_buffers[i];
            }
        }

        //MPI_Gather(partition_data, this->partition_height_ * this->partition_width_, MPI_DOUBLE, matrix, this->partition_height_ * this->partition_width_, MPI_DOUBLE, 0, this->GetCartesianCommunicator());
        delete[] proc_buffers;
        return matrix;
    }

    double *Matrix::AssemblePartition()
    {
        double *partition_data = new double[this->partition_width_ * this->partition_height_];
        for (int row = 0; row < this->partition_height_; row++)
        {
            partition_data[row * this->partition_width_] = this->left_border_[row];
            std::copy(
                this->inner_points_ + (row + 1) * this->partition_width_ + 1,
                this->inner_points_ + (row + 1) * this->partition_width_ + this->partition_width_ - 1,
                partition_data + row * this->partition_width_ + 1);
            partition_data[row * this->partition_width_ + (this->partition_width_ - 1)] = this->right_border_[row];
        }

        return partition_data;
    }
    void Matrix::GetPartitionCoordinates(int rank, int *partition_coords)
    {

        MPI_Cart_coords(this->GetCartesianCommunicator(), rank, 2, partition_coords);
    }
    void Matrix::PrintPartitionData()
    {
        std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: (" << this->partition_coords_[0] << ", " << this->partition_coords_[1] << "); Top ID: " << this->neighbours[0].GetNeighborId() << "; Right ID: " << this->neighbours[1].GetNeighborId() << "; Bottom ID: " << this->neighbours[2].GetNeighborId() << "; Left ID: " << this->neighbours[3].GetNeighborId() << "; Partition size: " << this->partition_width_ << "x" << this->partition_height_ << std::endl;
        printf("       ");
        if (this->IsTopBorder())
        {
            for (int i = 0; i < this->partition_width_; i++)
            {
                printf("%6.2f ", -1.0);
            }
        }
        else
        {
            for (int i = 0; i < this->partition_width_; i++)
            {
                printf("%6.2f ", this->top_ghost[i]);
            }
        }
        printf("       \n");
        for (int i = 0; i < this->partition_height_; i++)
        {
            if (this->IsLeftBorder())
            {
                printf("%6.2f ", -1.0);
            }
            else
            {
                printf("%6.2f ", this->left_ghost_points_[i]);
            }

            printf("%6.2f ", this->left_border_[i]);

            int offset = (i + 1) * (this->partition_width_); // Add one to i because the first row of the inner data is the top ghost.
            for (int j = 1; j < this->partition_width_ - 1; j++)
            {
                printf("%6.2f ", this->inner_points_[offset + j]);
            }

            printf("%6.2f ", this->right_border_[i]);

            if (this->IsRightBorder())
            {
                printf("%6.2f ", -1.0);
            }
            else
            {
                printf("%6.2f ", this->right_ghost_points_[i]);
            }
            printf("\n");
        }

        printf("       ");
        if (this->IsBottomBorder())
        {
            for (int col = 0; col < this->partition_width_; col++)
            {
                printf("%6.2f ", -1.0);
            }
        }
        else
        {
            for (int col = 0; col < this->partition_width_; col++)
            {
                //printf("%6.2f ", this->inner_points_[(this->partition_height_ + 1) * this->partition_width_ + col]);
                printf("%6.2f ", this->bottom_ghost[col]);
            }
        }
        printf("       \n");
    }

    void Matrix::PrintAllPartitions()
    {
        for (int i = 0; i < this->proc_count_; i++)
        {
            MPI_Barrier(this->GetCartesianCommunicator());
            if (i == this->proc_id_)
            {
                this->PrintPartitionData();
                fflush(stdout);
            }
        }

        MPI_Barrier(this->GetCartesianCommunicator());
    }

    bool Matrix::IsTopBorder()
    {
        return this->partition_coords_[0] == this->top_grid_border_coord;
    }
    bool Matrix::IsBottomBorder()
    {
        return this->partition_coords_[0] == this->bottom_grid_border_coord;
    }
    bool Matrix::IsLeftBorder()
    {
        return this->partition_coords_[1] == this->left_grid_border_coord;
    }
    bool Matrix::IsRightBorder()
    {
        return this->partition_coords_[1] == this->right_grid_border_coord;
    }

} // namespace pde_solver::data::cpu_distr
