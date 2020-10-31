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
namespace pde_solver
{
    Matrix::Matrix(int halo_size, int width, int height) : pde_solver::common::BaseMatrix(width, height)
    {
        this->halo_size_ = halo_size;
        this->is_initialized_ = false;
        for (int i = 0; i < 2; i++)
        {
            this->processes_per_dimension_[i] = 0;
        }
    }

    Matrix::Matrix(MatrixConfiguration config) : pde_solver::common::BaseMatrix(config.n_cols, config.n_rows)
    {
        this->matrix_config_ = config;
        this->halo_size_ = 1;
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

    Matrix Matrix::CloneShell()
    {
        Matrix m(this->halo_size_, this->matrix_width_, this->matrix_height_);
        m.neighbours = this->neighbours;
        m.partition_coords_[0] = this->partition_coords_[0];
        m.partition_coords_[1] = this->partition_coords_[1];
        m.processes_per_dimension_[0] = this->processes_per_dimension_[0];
        m.processes_per_dimension_[1] = this->processes_per_dimension_[1];
        m.partition_height_ = this->partition_height_;
        m.partition_width_ = this->partition_width_;
        m.proc_id_ = this->proc_id_;
        m.proc_count_ = this->proc_count_;
        m.x_partitions_ = this->x_partitions_;
        m.y_partitions_ = this->y_partitions_;
        m.is_initialized_ = this->is_initialized_;
        m.right_grid_border_coord_ = this->right_grid_border_coord_;
        m.left_grid_border_coord_ = this->left_grid_border_coord_;
        m.bottom_grid_border_coord_ = this->bottom_grid_border_coord_;
        m.top_grid_border_coord_ = this->top_grid_border_coord_;
        m.cartesian_communicator_ = this->cartesian_communicator_;
        m.InitData(this->initial_inner_value_, this->initial_left_value_, this->initial_right_value_, this->initial_bottom_value_, this->initial_top_value_);
        return m;
    }

    void Matrix::InitData(double inner_value, double left_border_value, double right_border_value, double bottom_border_value, double top_border_value)
    {
        this->initial_inner_value_ = inner_value;
        this->initial_left_value_ = left_border_value;
        this->initial_right_value_ = right_border_value;
        this->initial_bottom_value_ = bottom_border_value;
        this->initial_top_value_ = top_border_value;
        int inner_data_width = this->partition_width_;       //(this->partition_width_ - 2);
        int inner_data_height = this->partition_height_ + 2; // Two extra rows to hold the top and bottom halo
                                                             //this->inner_points_ = new double[inner_data_height * inner_data_width]; // Since the left and right borders of the partitions are stored separately, there is no need to store them on teh inner points array.
        this->AllocateMemory();
        this->InitLeftBorderAndGhost(inner_value, left_border_value, bottom_border_value, top_border_value);
        this->InitRightBorderAndGhost(inner_value, right_border_value, bottom_border_value, top_border_value);

        // init top row. This will be the top ghost points which come from the bottom border of the partition above it.

        int row = 0;
        int col;

        // Top halo initialization
        // If the partition is on top border of the grid, there is no top halo to initialize.

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
    }

    void Matrix::Init(double value, int argc, char *argv[])
    {
        this->Init(value, value, value, value, value, argc, argv);
    }

    void Matrix::Deallocate()
    {
        delete[] left_ghost_points_;
        delete[] right_ghost_points_;
        delete[] left_border_;
        delete[] right_border_;
        delete[] inner_points_;
    }

    void Matrix::AllocateMemory()
    {
        int inner_data_width = this->partition_width_;                          //(this->partition_width_ - 2);
        int inner_data_height = this->partition_height_ + 2;                    // Two extra rows to hold the top and bottom halo
        this->inner_points_ = new double[inner_data_height * inner_data_width]; // Since the left and right borders of the partitions are stored separately, there is no need to store them on teh inner points array.
        this->left_border_ = new double[this->partition_height_];
        this->left_ghost_points_ = new double[this->partition_height_];
        this->right_border_ = new double[this->partition_height_];
        this->right_ghost_points_ = new double[this->partition_height_];
        this->top_ghost = &this->inner_points_[0];
        this->bottom_ghost = &this->inner_points_[(inner_data_height - 1) * inner_data_width];
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
        this->InitData(inner_value, left_border_value, right_border_value, bottom_border_value, top_border_value);

        MPI_Barrier(this->GetCartesianCommunicator());
    }

    void Matrix::InitLeftBorderAndGhost(double inner_value, double left_border_value, double bottom_border_value, double top_border_value)
    {

        // assing the left border values. If the partition is on the border of the cartesian grid,
        // the left border values should receive the values from the parameter 'left_border_value'

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
            // int deviceCount = 0;
            // printf("Number of GPUs: %d\n", deviceCount);
        }
        MPI_Barrier(this->GetCartesianCommunicator());
        std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: (" << this->partition_coords_[0] << ", " << this->partition_coords_[1] << "); Top ID: " << this->neighbours[0].GetNeighborId() << "; Right ID: " << this->neighbours[1].GetNeighborId() << "; Bottom ID: " << this->neighbours[2].GetNeighborId() << "; Left ID: " << this->neighbours[3].GetNeighborId() << "Partition size: " << this->partition_width_ << "x" << this->partition_height_ << std::endl;
        MPI_Barrier(this->GetCartesianCommunicator());
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
        this->right_grid_border_coord_ = this->processes_per_dimension_[1] - 1;
        this->bottom_grid_border_coord_ = this->processes_per_dimension_[0] - 1;
        this->left_grid_border_coord_ = 0;
        this->top_grid_border_coord_ = 0;

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
        Deallocate();
        MPI_Barrier(this->GetCartesianCommunicator());
        int result = MPI_Finalize();
        if (this->proc_id_ == 0)
        {
            printf("MPi Finalize, process %d, return code: %d\n", this->proc_id_, result);
        }
    }

    double Matrix::LocalSweep(Matrix new_matrix)
    {
        this->current_max_difference_ = 0;
        double diff, new_value;

        int max_thread_num = omp_get_max_threads();

        // 4 sends and 4 receives
        MPI_Request requests[8];

        // Two levels of parallelization
        omp_set_nested(2);

#pragma omp parallel sections firstprivate(diff, new_value, max_thread_num) num_threads(std::min(max_thread_num, 4)) reduction(max \
                                                                                                                               : current_max_difference_)
        {
#pragma omp section
            {
                if (!this->IsLeftBorder())
                {
                    auto thread_id = omp_get_thread_num();
                    if (!this->IsTopBorder())
                    {
                        new_value = (this->left_ghost_points_[0] + this->top_ghost[0] + this->GetLocal(0, 1) + this->GetLocal(1, 0)) / 4;
                        diff = fabs(new_value - this->GetLocal(0, 0));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, 0, 0);
                    }

#pragma omp parallel for reduction(max \
                                   : current_max_difference_)
                    for (int i = 1; i < this->partition_height_ - 1; i++)
                    {
                        new_value = (this->GetLocal(i - 1, 0) + this->GetLocal(i + 1, 0) + this->left_ghost_points_[i] + this->GetLocal(i, 1)) / 4;
                        diff = fabs(new_value - this->GetLocal(i, 0));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, i, 0);
                    }

                    if (!this->IsBottomBorder())
                    {
                        new_value = (this->left_ghost_points_[this->partition_height_ - 1] + this->bottom_ghost[0] + this->GetLocal(this->partition_height_ - 2, 0) + this->GetLocal(this->partition_height_ - 1, 1)) / 4;
                        diff = fabs(new_value - this->GetLocal(this->partition_height_ - 1, 0));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, this->partition_height_ - 1, 0);
                    }
                    auto neighbour_left = this->GetNeighbour(PartitionNeighbourType::LEFT_NEIGHTBOUR);

                    MPI_Isend(new_matrix.left_border_, this->partition_height_, MPI_DOUBLE, neighbour_left.id, 0, this->GetCartesianCommunicator(), &requests[0]);
                    MPI_Irecv(new_matrix.left_ghost_points_, this->partition_height_, MPI_DOUBLE, neighbour_left.id, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[4]);
                }
                else
                {
                    MPI_Isend(this->left_border_, this->partition_height_, MPI_DOUBLE, MPI_PROC_NULL, 0, this->GetCartesianCommunicator(), &requests[0]);
                    MPI_Irecv(new_matrix.left_ghost_points_, this->partition_height_, MPI_DOUBLE, MPI_PROC_NULL, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[4]);
                }
            }

#pragma omp section
            {

                if (!this->IsRightBorder())
                {
                    if (!this->IsTopBorder())
                    {
                        new_value = (this->right_ghost_points_[0] + this->top_ghost[this->partition_width_ - 1] + this->GetLocal(0, this->partition_width_ - 2) + this->GetLocal(1, this->partition_width_ - 1)) / 4;
                        diff = fabs(new_value - this->GetLocal(0, this->partition_width_ - 1));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, 0, this->partition_width_ - 1);
                    }

#pragma omp parallel for reduction(max \
                                   : current_max_difference_)
                    for (int i = 1; i < this->partition_height_ - 1; i++)
                    {
                        new_value = (this->GetLocal(i, this->partition_width_ - 2) + this->right_ghost_points_[i] + this->GetLocal(i - 1, this->partition_width_ - 1) + this->GetLocal(i + 1, this->partition_width_ - 1)) / 4;
                        diff = fabs(new_value - this->GetLocal(i, this->partition_width_ - 1));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }

                        new_matrix.SetLocal(new_value, i, this->partition_width_ - 1);
                    }

                    if (!this->IsBottomBorder())
                    {
                        new_value = (this->right_ghost_points_[this->partition_height_ - 1] + this->bottom_ghost[this->partition_width_ - 1] + this->GetLocal(this->partition_height_ - 1, this->partition_width_ - 2) + this->GetLocal(this->partition_height_ - 2, this->partition_width_ - 1)) / 4;
                        diff = fabs(new_value - this->GetLocal(this->partition_height_ - 1, this->partition_width_ - 1));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, this->partition_height_ - 1, this->partition_width_ - 1);
                    }
                    auto neighbour_right = this->GetNeighbour(PartitionNeighbourType::RIGHT_NEIGHBOUR);

                    MPI_Isend(new_matrix.right_border_, this->partition_height_, MPI_DOUBLE, neighbour_right.id, 0, this->GetCartesianCommunicator(), &requests[1]);
                    MPI_Irecv(new_matrix.right_ghost_points_, this->partition_height_, MPI_DOUBLE, neighbour_right.id, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[5]);
                }
                else
                {
                    MPI_Isend(new_matrix.right_border_, this->partition_height_, MPI_DOUBLE, MPI_PROC_NULL, 0, this->GetCartesianCommunicator(), &requests[1]);
                    MPI_Irecv(new_matrix.right_ghost_points_, this->partition_height_, MPI_DOUBLE, MPI_PROC_NULL, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[5]);
                }
            }

#pragma omp section
            {
                if (!this->IsTopBorder())
                {
#pragma omp parallel for reduction(max \
                                   : current_max_difference_)
                    for (int i = 1; i < this->partition_width_ - 1; i++)
                    {
                        new_value = (this->GetLocal(0, i - 1) + this->GetLocal(0, i + 1) + this->GetLocal(1, i) + this->top_ghost[i]) / 4;
                        diff = fabs(new_value - this->GetLocal(0, i));
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, 0, i);
                    }

                    auto neighbour_top = this->GetNeighbour(PartitionNeighbourType::TOP_NEIGHBOUR);
                    MPI_Isend(&new_matrix.inner_points_[this->partition_width_], this->partition_width_, MPI_DOUBLE, neighbour_top.id, 0, this->GetCartesianCommunicator(), &requests[2]);
                    MPI_Irecv(new_matrix.top_ghost, this->partition_width_, MPI_DOUBLE, neighbour_top.id, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[6]);
                }
                else
                {
                    MPI_Isend(&new_matrix.inner_points_[this->partition_width_], this->partition_width_, MPI_DOUBLE, MPI_PROC_NULL, 0, this->GetCartesianCommunicator(), &requests[2]);
                    MPI_Irecv(new_matrix.top_ghost, this->partition_width_, MPI_DOUBLE, MPI_PROC_NULL, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[6]);
                }
            }

#pragma omp section
            {
                if (!this->IsBottomBorder())
                {
                    //new_matrix.SetLocal(new_value, this->partition_height_ - 1, 1);

#pragma omp parallel for reduction(max \
                                   : current_max_difference_)
                    for (int i = 1; i < this->partition_width_ - 1; i++)
                    {
                        new_value = (this->GetLocal(this->partition_height_ - 1, i - 1) + this->GetLocal(this->partition_height_ - 1, i + 1) + this->GetLocal(this->partition_height_ - 2, i) + this->bottom_ghost[i]) / 4;
                        if (diff > current_max_difference_)
                        {
                            current_max_difference_ = diff;
                        }
                        new_matrix.SetLocal(new_value, this->partition_height_ - 1, i);
                    }

                    auto neighbour_bottom = this->GetNeighbour(PartitionNeighbourType::BOTTOM_NEIGHBOUR);
                    MPI_Isend(&new_matrix.inner_points_[this->partition_height_ * this->partition_width_], this->partition_width_, MPI_DOUBLE, neighbour_bottom.id, 0, this->GetCartesianCommunicator(), &requests[3]);
                    MPI_Irecv(new_matrix.bottom_ghost, this->partition_width_, MPI_DOUBLE, neighbour_bottom.id, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[7]);
                }
                else
                {
                    MPI_Isend(&new_matrix.inner_points_[this->partition_height_ * this->partition_width_], this->partition_width_, MPI_DOUBLE, MPI_PROC_NULL, 0, this->GetCartesianCommunicator(), &requests[3]);
                    MPI_Irecv(new_matrix.bottom_ghost, this->partition_width_, MPI_DOUBLE, MPI_PROC_NULL, MPI_ANY_TAG, this->GetCartesianCommunicator(), &requests[7]);
                }
            }
        }

#pragma omp parallel reduction(max \
                               : current_max_difference_)
        {
            int i, j;
#pragma omp for collapse(2) private(i, j, diff, new_value)
            for (i = 1; i < this->partition_height_ - 1; i++)
                for (j = 1; j < this->partition_width_ - 1; j++)
                {
                    new_value = (this->GetLocal(i - 1, j) + this->GetLocal(i + 1, j) + this->GetLocal(i, j - 1) + this->GetLocal(i, j + 1)) / 4;
                    diff = fabs(new_value - this->GetLocal(i, j));
                    if (diff > current_max_difference_)
                    {
                        current_max_difference_ = diff;
                    }
                    new_matrix.SetLocal(new_value, i, j);
                }
        }
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        return current_max_difference_;
    }

    const PartitionNeighbour Matrix::GetNeighbour(PartitionNeighbourType neighbour_type)
    {
        switch (neighbour_type)
        {
        case PartitionNeighbourType::TOP_NEIGHBOUR:
            return this->neighbours[0];
        case PartitionNeighbourType::BOTTOM_NEIGHBOUR:
            return this->neighbours[2];
        case PartitionNeighbourType::LEFT_NEIGHTBOUR:
            return this->neighbours[3];
        case PartitionNeighbourType::RIGHT_NEIGHBOUR:
            return this->neighbours[1];

        default:
            throw new std::out_of_range("The neighbour type must be TOP_NEIGHBOUR, BOTTOM_NEIGHBOUR, RIGHT_NEIGHBOUR or LEFT_NEIGHBOUR");
        }
    }

    const double Matrix::GetLocal(int partition_row, int partition_col)
    {
        if (partition_col < 0 || partition_col > this->partition_width_ - 1 || partition_row < 0 || partition_row > this->partition_height_ - 1)
        {
            throw new std::out_of_range("Index is out of range for the partition");
        }
        if (partition_col == 0)
        {
            return this->left_border_[partition_row];
        }

        if (partition_col == this->partition_width_ - 1)
        {
            return right_border_[partition_row];
        }

        int row_offset = (partition_row + 1) * this->partition_width_;

        return this->inner_points_[row_offset + partition_col];
    }

    void Matrix::SetLocal(double value, int row, int col)
    {
        if (col == 0)
        {
            this->left_border_[row] = value;
            // If it's the left border on the first or last row of the partition, copy it to the inner points as well because it will be easier to transfer it to the bottom and top ghost points of the top and bottom neightbours.
            if (row == 0 || row == this->partition_height_ - 1)
            {
                this->inner_points_[(row + 1) * this->partition_width_] = value;
            }

            return;
        }

        if (col == this->partition_width_ - 1)
        {
            this->right_border_[row] = value;
            // If it's the left border on the first or last row of the partition, copy it to the inner points as well because it will be easier to transfer it to the bottom and top ghost points of the top and bottom neightbours.
            if (row == 0 || row == this->partition_height_ - 1)
            {
                this->inner_points_[(row + 1) * this->partition_width_ + col] = value;
            }

            return;
        }

        this->inner_points_[(row + 1) * this->partition_width_ + col] = value;
    }

    void Matrix::SetGlobal(double value, int row, int col)
    {
        if (row < 0 || col < 0)
        {
            return;
        }

        if (row > this->matrix_height_ - 1)
        {
            return;
        }

        if (col > this->matrix_width_ - 1)
        {
            return;
        }

        int partition_row_start = this->partition_coords_[0] * this->partition_height_;
        int partition_row_end = this->partition_coords_[0] * this->partition_height_ + this->partition_height_ - 1;
        int partition_col_start = this->partition_coords_[1] * this->partition_width_;
        int partition_col_end = this->partition_coords_[1] * this->partition_width_ + this->partition_width_ - 1;

        if (row < partition_row_start || row > partition_row_end)
        {
            return;
        }

        if (col < partition_col_start || col > partition_col_end)
        {
            return;
        }
        int local_row = row - partition_row_start;
        int local_col = col - partition_col_start;
        this->SetLocal(value, local_row, local_col);
    }

    const double Matrix::GlobalDifference()
    {
        MPI_Barrier(this->GetCartesianCommunicator());
        MPI_Allreduce(&current_max_difference_, &current_max_difference_, 1, MPI_DOUBLE, MPI_MAX, this->GetCartesianCommunicator());
        return current_max_difference_;
    }

    void Matrix::ShowMatrix()
    {
        double *partition_data = this->AssemblePartition();
        double *buffer, *matrix;
        if (this->proc_id_ == 0)
        {
            buffer = new double[this->matrix_height_ * this->matrix_width_];
            matrix = new double[this->matrix_height_ * this->matrix_width_];
        }

        MPI_Gather(partition_data, this->partition_height_ * this->partition_width_, MPI_DOUBLE, buffer, this->partition_height_ * this->partition_width_, MPI_DOUBLE, 0, this->GetCartesianCommunicator());
        delete[] partition_data;
        if (this->proc_id_ != 0)
        {
            return;
        }

        for (int i = 0; i < this->proc_count_; i++)
        {

            int coords[2];
            this->GetPartitionCoordinates(i, coords);
            for (int row = 0; row < this->partition_height_; row++)
            {
                int matrix_row = coords[0] * this->partition_height_ + row;
                int start_in_row = coords[1] * partition_width_;
                int start_of_this_process = i * this->partition_width_ * this->partition_height_;
                std::copy(buffer + start_of_this_process + row * this->partition_width_, buffer + start_of_this_process + row * this->partition_width_ + this->partition_width_, matrix + (matrix_row * this->matrix_width_) + start_in_row);
            }
        }

        delete[] buffer;
        for (int i = 0; i < this->matrix_height_; i++)
        {
            for (int j = 0; j < this->matrix_width_; j++)
            {
                printf("%6.2f ", matrix[i * this->matrix_width_ + j]);
            }
            putchar('\n');
        }

        delete[] matrix;
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

    void Matrix::Synchronize()
    {
        MPI_Barrier(this->GetCartesianCommunicator());
    }

    void Matrix::GetPartitionCoordinates(int rank, int *partition_coords)
    {

        MPI_Cart_coords(this->GetCartesianCommunicator(), rank, 2, partition_coords);
    }
    void Matrix::PrintPartitionData()
    {
        std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: (" << this->partition_coords_[0] << ", " << this->partition_coords_[1] << "); Top ID: " << this->neighbours[0].GetNeighborId() << "; Right ID: " << this->neighbours[1].GetNeighborId() << "; Bottom ID: " << this->neighbours[2].GetNeighborId() << "; Left ID: " << this->neighbours[3].GetNeighborId() << "; Partition size: " << this->partition_width_ << "x" << this->partition_height_ << std::endl;
        double *partition_data = this->AssemblePartition();
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

            int offset = i * (this->partition_width_);
            for (int j = 0; j < this->partition_width_; j++)
            {
                printf("%6.2f ", partition_data[offset + j]);
            }

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
    }

    bool Matrix::IsTopBorder()
    {
        return this->partition_coords_[0] == this->top_grid_border_coord_;
    }
    bool Matrix::IsBottomBorder()
    {
        return this->partition_coords_[0] == this->bottom_grid_border_coord_;
    }
    bool Matrix::IsLeftBorder()
    {
        return this->partition_coords_[1] == this->left_grid_border_coord_;
    }
    bool Matrix::IsRightBorder()
    {
        return this->partition_coords_[1] == this->right_grid_border_coord_;
    }
} // namespace pde_solver
