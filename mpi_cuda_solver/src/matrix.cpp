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
    Matrix::Matrix(int halo_size, int width, int height) : pde_solver::BaseMatrix(width, height)
    {
        this->halo_size_ = halo_size;
        this->is_initialized_ = false;
        for (int i = 0; i < 2; i++)
        {
            this->processes_per_dimension_[i] = 0;
        }
    }

    Matrix::Matrix(MatrixConfiguration config) : pde_solver::BaseMatrix(config.n_cols, config.n_rows)
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

    void Matrix::InitData(float inner_value, float left_border_value, float right_border_value, float bottom_border_value, float top_border_value)
    {
        this->initial_inner_value_ = inner_value;
        this->initial_left_value_ = left_border_value;
        this->initial_right_value_ = right_border_value;
        this->initial_bottom_value_ = bottom_border_value;
        this->initial_top_value_ = top_border_value;
        int inner_data_width = this->partition_width_;       //(this->partition_width_ - 2);
        int inner_data_height = this->partition_height_ + 2; // Two extra rows to hold the top and bottom halo
                                                             //this->inner_points_ = new float[inner_data_height * inner_data_width]; // Since the left and right borders of the partitions are stored separately, there is no need to store them on teh inner points array.
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
                // No need to assign the part that will go to the GPU because the calculation will not be done.
            }
            else
            {
                this->inner_points_[0] = inner_value;
                this->left_region[this->partition_height_ + 2] = inner_value; // first row, second column.
            }

            if (this->IsRightBorder())
            {
                // the value on the last column of the halo will come from the right border
                this->inner_points_[inner_data_width - 1] = right_border_value;
            }
            else
            {
                this->inner_points_[inner_data_width - 1] = inner_value;
                this->right_region[this->partition_height_ + 2] = inner_value; // first row, second column.
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
                this->left_region[this->partition_height_ + 2 + this->partition_height_ + 1] = inner_value; // last row, second column.
            }

            if (this->IsRightBorder())
            {
                // the value on the last column of the halo will come from the right border
                this->inner_points_[inner_data_width * inner_data_height - 1] = right_border_value;
            }
            else
            {
                this->inner_points_[inner_data_width * inner_data_height - 1] = inner_value;
                this->right_region[this->partition_height_ + 2 + this->partition_height_ + 1] = inner_value; // last row, second column.
            }

            for (col = 1; col < inner_data_width - 1; col++)
            {
                this->inner_points_[row * inner_data_width + col] = inner_value;
            }
        }

        // Inner data
        for (int i = 1; i < inner_data_height - 1; i++)
        {
            for (int j = 1; j < inner_data_width - 1; j++)
            {
                if (this->IsTopBorder() && i == 1)
                {
                    this->SetLocal(top_border_value, i, j);
                }
                else if (this->IsBottomBorder() && i == (inner_data_height - 2))
                {
                    this->SetLocal(bottom_border_value, i, j);
                }
                else
                {
                    this->SetLocal(inner_value, i, j);
                }
            }
        }

        // Move data to GPU
        for (int i = 0; i < this->matrix_config_.gpu_number; i++)
        {
            float *start = &(this->inner_points_[this->inner_data_streams[i].gpu_region_start * (this->inner_data_streams[i].gpu_data_width + 2)]);
            float *start_left_border = &(this->left_border_[this->inner_data_streams[i].halo_points_host_start]);
            float *start_right_border = &(this->right_border_[this->inner_data_streams[i].halo_points_host_start]);
            cudaSetDevice(this->inner_data_streams[i].gpu_id);
            cudaMemcpyAsync(this->inner_data_streams[i].d_data, start, (this->inner_data_streams[i].gpu_data_width) * (this->inner_data_streams[i].gpu_data_height + 2) * sizeof(float), cudaMemcpyHostToDevice, this->inner_data_streams[i].stream);
            cudaMemcpyAsync(this->inner_data_streams[i].halo_points, start_left_border, sizeof(float) * (this->inner_data_streams[i].gpu_data_height), cudaMemcpyHostToDevice, this->inner_data_streams[i].stream);
            cudaMemcpyAsync((this->inner_data_streams[i].halo_points + ), start_right_border, sizeof(float) * (this->inner_data_streams[i].gpu_data_height), cudaMemcpyHostToDevice, this->inner_data_streams[i].stream);
        }

        // for (int i = 0; i < 4; i++)
        // {
        //     if ((i == 0 && !this->IsLeftBorder()) ||
        //         (i == 2 && !this->IsTopBorder()) ||
        //         (i == 2 && !this->IsRightBorder()) ||
        //         (i == 2 && !this->IsBottomBorder()))
        //     {
        //         switch (i)
        //         {
        //         case 0:

        //             /* code */
        //             break;
        //         case 1:
        //             break;
        //         case 2:
        //             break;
        //         case 3:
        //             break;
        //         default:
        //             break;
        //         }
        //     }
        // }
    }

    void Matrix::Init(float value, int argc, char *argv[])
    {
        this->Init(value, value, value, value, value, argc, argv);
    }

    void Matrix::ConfigGpuExecution()
    {
        // there must be one stream per border. Each stream can be in one GPU or one GPU can have multiple streams in case the
        int current_device = 0;
        for (int i = 0; i < 4; i++)
        {

            cudaSetDevice(current_device);
            border_calc_streams[i].gpu_id = current_device;
            cudaStreamCreate(&(border_calc_streams[i].stream));

            // The stream is created for a border that is NOT also a matrix border which means that the calculation will be performed.
            // In this case, if possible, the device should be changed.
            if ((i == 0 && !this->IsLeftBorder()) ||
                (i == 1 && !this->IsTopBorder()) ||
                (i == 2 && !this->IsRightBorder()) ||
                (i == 3 && !this->IsBottomBorder()))
            {
                current_device++;
                if (current_device == this->matrix_config_.gpu_number)
                {
                    // Return back to the first device if we were on the last device.
                    current_device = 0;
                }
            }
        }

        this->inner_data_streams = new GPUStream[this->matrix_config_.gpu_number];
        // configure inner data streams
        for (int i = 0; i < this->matrix_config_.gpu_number; i++)
        {
            cudaSetDevice(i);
            cudaStreamCreate(&(this->inner_data_streams[i].stream));
            this->inner_data_streams[i].gpu_id = i;
        }
    }

    void Matrix::Deallocate()
    {
        cudaFreeHost(&left_region);
        cudaFreeHost(&right_region);

        cudaFreeHost(&inner_points_);

        for (int i = 0; i < 4; i++)
        {
            cudaFree(this->border_calc_streams[i].d_data);
        }

        for (int i = 0; i < this->matrix_config_.gpu_number; i++)
        {
            cudaFree(this->inner_data_streams[i].d_data);
        }
    }

    int calc_gpu_rows(int total_rows, float gpu_perc)
    {
        return ceil(total_rows * gpu_perc);
    }

    void Matrix::AllocateMemory()
    {
        int inner_data_width = this->partition_width_;       //(this->partition_width_ - 2);
        int inner_data_height = this->partition_height_ + 2; // Two extra rows to hold the top and bottom halo
                                                             // this->inner_points_ = new float[inner_data_height * inner_data_width]; // Since the left and right borders of the partitions are stored separately, there is no need to store them on teh inner points array.
        int side_region_height = this->partition_height_ + 2 * this->halo_size_;
        int side_region_width = 1 + 2 * this->halo_size_;
        cudaMallocHost(&(this->inner_points_), inner_data_height * inner_data_width * sizeof(float));
        cudaMallocHost(&(this->left_region), side_region_height * side_region_width * sizeof(float));
        cudaMallocHost(&(this->right_region), side_region_height * side_region_width * sizeof(float));

        this->left_border_ = &(this->left_region[side_region_height + 1]);
        this->left_ghost_points_ = &(this->left_region[1]);
        this->left_border_inner_halo = &(this->left_region[2 * side_region_height + 1]);
        this->right_border_ = &(this->right_region[side_region_height + 1]);
        this->right_ghost_points_ = &(this->right_region[1]);
        this->right_border_inner_halo = &(this->right_region[2 * side_region_height + 1]);
        this->top_ghost = &this->inner_points_[0];
        this->bottom_ghost = &this->inner_points_[(inner_data_height - 1) * inner_data_width];

        //float *inner_points;

        int inner_rows = this->partition_height_ - 2;

        int gpu_rows = calc_gpu_rows(inner_rows, 1 - this->matrix_config_.cpu_perc);

        int rows_per_gpu = floor(((float)gpu_rows) / this->matrix_config_.gpu_number);
        int rows_allocated = 0;
        int inner_data_gpu_width = this->partition_width_ - 2; // the two borders will be on a separate container
        //cudaError err;
        for (int i = 0; i < this->matrix_config_.gpu_number - 1; i++)
        {
            cudaSetDevice(this->inner_data_streams[i].gpu_id);
            CUDA_CHECK_RETURN(cudaMalloc(&(this->inner_data_streams[i].d_data), (rows_per_gpu + 2) * (inner_data_width) * sizeof(float)));
            rows_allocated += rows_per_gpu;
            CUDA_CHECK_RETURN(cudaMalloc(&(this->inner_data_streams[i].halo_points), (rows_per_gpu * 2) * sizeof(float)));

            this->inner_data_streams[i].gpu_region_start = (inner_rows - rows_allocated) + 2;                      // add 2 to offset the top halo and the top border that are stored on the inner_data variable
            this->inner_data_streams[i].halo_points_host_start = this->inner_data_streams[i].gpu_region_start - 1; // the left border and right border have 2 rows less than the inner data.
            this->inner_data_streams[i].gpu_data_width = this->partition_width_;
            this->inner_data_streams[i].gpu_data_height = rows_per_gpu;
            this->inner_data_streams[i].is_contiguous_on_host = true;
        }

        cudaSetDevice(this->inner_data_streams[this->matrix_config_.gpu_number - 1].gpu_id);
        int remaining_gpu_rows = (gpu_rows - rows_allocated);
        rows_allocated += remaining_gpu_rows;

        CUDA_CHECK_RETURN(cudaMalloc(&(this->inner_data_streams[this->matrix_config_.gpu_number - 1].d_data), (remaining_gpu_rows + 2) * (inner_data_width) * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc(&(this->inner_data_streams[this->matrix_config_.gpu_number - 1].halo_points), (remaining_gpu_rows * 2) * sizeof(float)));
        this->inner_data_streams[this->matrix_config_.gpu_number - 1].gpu_region_start = (inner_rows - rows_allocated) + 2;                                                        // add 2 to offset the top halo and the top border that are stored on the inner_data variable
        this->inner_data_streams[this->matrix_config_.gpu_number - 1].halo_points_host_start = this->inner_data_streams[this->matrix_config_.gpu_number - 1].gpu_region_start - 1; // the side broders have two columns less than the
        this->inner_data_streams[this->matrix_config_.gpu_number - 1].gpu_data_width = this->partition_width_;
        this->inner_data_streams[this->matrix_config_.gpu_number - 1].gpu_data_height = remaining_gpu_rows;
        this->inner_data_streams[this->matrix_config_.gpu_number - 1].is_contiguous_on_host = true;

        for (int i = 0; i < 4; i++)
        {
            CUDA_CHECK_RETURN(cudaSetDevice(border_calc_streams[i].gpu_id));
            if ((i == 0 && !this->IsLeftBorder()) ||
                (i == 2 && !this->IsRightBorder()))
            {
                // remove 2 from partition height because the first and last element of the borders are always calculated by the CPU.
                int gpu_elements = calc_gpu_rows(this->partition_height_ - 2, this->matrix_config_.cpu_perc);
                gpu_elements += 2; // add two for the top and bottom halo cells.
                gpu_elements *= 3; // multiply by 3 because three columns will need to be stored. the right and left halo of each border needs to be stored.
                CUDA_CHECK_RETURN(cudaMalloc(&(border_calc_streams[i].d_data), gpu_elements * sizeof(float)));
                this->border_calc_streams[i].gpu_region_start = (this->partition_height_ - 2) - gpu_elements + 1; // Add one to offset the first element of the border which will be calculated by the CPU
            }
            if (
                (i == 1 && !this->IsTopBorder()) ||
                (i == 3 && !this->IsBottomBorder()))
            {
                // the top and bottom border are two elements smaller then the partition width. Additionally, the first and last element are calculated by the CPU.
                int gpu_elements = calc_gpu_rows(this->partition_width_ - 4, 1 - this->matrix_config_.cpu_perc);
                gpu_elements += 2; // add two for the top and bottom halo cells.
                gpu_elements *= 3; // multiply by 3 because three columns will need to be stored. the right and left halo of each border needs to be stored.
                CUDA_CHECK_RETURN(cudaMalloc(&(border_calc_streams[i].d_data), gpu_elements * sizeof(float)));
                this->border_calc_streams[i].gpu_region_start = (this->partition_width_ - 4) - gpu_elements + 2; // Add two to offset the first element of the row which belongs to the left border and the first element of the border whcih is calculated by the CPU.
            }
        }
    }

    void Matrix::Init(float inner_value, float left_border_value, float right_border_value, float bottom_border_value, float top_border_value, int argc, char *argv[])
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
        this->ConfigGpuExecution();
        this->InitData(inner_value, left_border_value, right_border_value, bottom_border_value, top_border_value);
        MPI_Barrier(this->GetCartesianCommunicator());
    }

    void Matrix::
        InitLeftBorderAndGhost(float inner_value, float left_border_value, float bottom_border_value, float top_border_value)
    {
        // assing the left border values. If the partition is on the border of the cartesian grid,
        // the left border values should receive the values from the parameter 'left_border_value'

        if (this->IsLeftBorder())
        {
            for (int i = 0; i < this->partition_height_; i++)
            {
                SetLocal(left_border_value, i, 0);
            }

            return;
        }

        int start = 0;
        if (this->IsTopBorder())
        {
            SetLocal(top_border_value, start, 0);
            this->left_ghost_points_[start] = top_border_value;
            start++;
        }

        int end = this->partition_height_ - 1;
        if (this->IsBottomBorder())
        {
            SetLocal(bottom_border_value, end, 0);
            this->left_ghost_points_[end] = bottom_border_value;
            end--;
        }
        for (int i = start; i <= end; i++)
        {
            SetLocal(inner_value, i, 0);
            this->left_ghost_points_[i] = inner_value;
        }
    }

    void Matrix::InitRightBorderAndGhost(float inner_value, float right_border_value, float bottom_border_value, float top_border_value)
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
            SetLocal(top_border_value, start, 0);
            this->right_ghost_points_[start] = top_border_value;
            start++;
        }

        int end = this->partition_height_ - 1;
        if (this->IsBottomBorder())
        {
            SetLocal(bottom_border_value, end, 0);
            this->right_ghost_points_[end] = bottom_border_value;
            end--;
        }
        for (int i = start; i <= end; i++)
        {
            SetLocal(inner_value, i, 0);
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

    float Matrix::LocalSweep(Matrix new_matrix, ExecutionStats *execution_stats)
    {
        for (int i = 0; i < this->matrix_config_.gpu_number; i++)
        {
            cudaSetDevice(this->inner_data_streams[i].gpu_id);
            cudaStreamSynchronize(this->inner_data_streams[i].stream);
        }

        for (int i = 0; i < 4; i++)
        {
            cudaSetDevice(this->border_calc_streams[i].gpu_id);
            cudaStreamSynchronize(this->border_calc_streams[i].stream);
        }

        this->current_max_difference_ = 0;
        float diff, new_value;

        int max_thread_num = omp_get_max_threads();

        // 4 sends and 4 receives
        MPI_Request requests[8];

        // Two levels of parallelization
        omp_set_nested(2);
        auto sweep_start = MPI_Wtime();
        auto border_start = MPI_Wtime();
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

                    // int gpu_start = (int)ceil((float)this->partition_height_ * this->matrix_config_.cpu_perc) + 1;
                    // int gpu_end = this->partition_height_ - 2; // last index of the elements to be processed by the GPU.
                    //cudaStream_t streams[this->matrix_config_.gpu_number];
                    // if (gpu_start <= gpu_end)
                    // {
                    //     // TODO (endizhupani@uni-muenster.de): Start the GPU Computation
                    // }

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
        execution_stats->total_border_calc_time += (MPI_Wtime() - border_start);

        auto inner_points_time = MPI_Wtime();

        // TODO (endizhupani@uni-muenster.de): Start the inner points calculation on the GPU.

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

        execution_stats->total_inner_points_time += (MPI_Wtime() - inner_points_time);

        auto idle_start = MPI_Wtime();
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        execution_stats->total_idle_comm_time += (MPI_Wtime() - idle_start);
        execution_stats->total_sweep_time += (MPI_Wtime() - sweep_start);
        execution_stats->n_sweeps += 1;
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

    const float Matrix::GetLocal(int partition_row, int partition_col)
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

    void Matrix::SetLocal(float value, int row, int col)
    {
        if (col == 0)
        {
            this->left_border_[row] = value;
            //// If it's the left border on the first or last row of the partition, copy it to the inner points as well because it will be easier to transfer it to the bottom and top ghost points of the top and bottom neightbours.
            // if (row == 0 || row == this->partition_height_ - 1)
            // {
            this->inner_points_[(row + 1) * this->partition_width_] = value; // Copy it to the inner data becuase it makes it mutch simpler to be processed by the GPU
            //}

            return;
        }

        if (col == this->partition_width_ - 1)
        {
            this->right_border_[row] = value;
            // // If it's the left border on the first or last row of the partition, copy it to the inner points as well because it will be easier to transfer it to the bottom and top ghost points of the top and bottom neightbours.
            // if (row == 0 || row == this->partition_height_ - 1)
            // {
            this->inner_points_[(row + 1) * this->partition_width_ + col] = value;
            // }

            return;
        }

        if (col == 1)
        {
            this->left_border_inner_halo[row] = value;
        }

        if (col == this->partition_width_ - 2)
        {
            this->left_border_inner_halo[row] = value;
        }

        this->inner_points_[(row + 1) * this->partition_width_ + col] = value;
    }

    void Matrix::SetGlobal(float value, int row, int col)
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

    const float Matrix::GlobalDifference(ExecutionStats *execution_stats)
    {
        MPI_Barrier(this->GetCartesianCommunicator());
        auto reduction_start = MPI_Wtime();
        float received_difference = 0;
        MPI_Allreduce(&current_max_difference_, &received_difference, 1, MPI_DOUBLE, MPI_MAX, this->GetCartesianCommunicator());
        current_max_difference_ = received_difference;
        execution_stats->total_time_reducing_difference += (MPI_Wtime() - reduction_start);
        execution_stats->n_diff_reducions += 1;
        return current_max_difference_;
    }

    void Matrix::ShowMatrix()
    {
        float *partition_data = this->AssemblePartition();
        float *buffer, *matrix;
        if (this->proc_id_ == 0)
        {
            buffer = new float[this->matrix_height_ * this->matrix_width_];
            matrix = new float[this->matrix_height_ * this->matrix_width_];
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

    float *Matrix::AssemblePartition()
    {
        float *partition_data = new float[this->partition_width_ * this->partition_height_];
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
        float *partition_data = this->AssemblePartition();
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
