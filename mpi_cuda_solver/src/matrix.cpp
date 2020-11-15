/**
 * matrix.cpp
 * Implementation for the classes defined in /include/matrix.h
 * ------------------------------------------------------
 * Copyright (c) 2020 Endi Zhupani
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "matrix.h"

namespace pde_solver {
Matrix::Matrix(int halo_size, int width, int height)
    : pde_solver::BaseMatrix(width, height) {
  this->halo_size_ = halo_size;
  this->is_initialized_ = false;
  this->processes_per_dimension_[0] = 0;

  // the number of processes along the horizontal dimension should always be
  // equal to 1.
  this->processes_per_dimension_[1] = 1;
}

Matrix::Matrix(MatrixConfiguration config)
    : pde_solver::BaseMatrix(config.n_cols, config.n_rows) {
  this->matrix_config_ = config;
  this->halo_size_ = 1;
  this->is_initialized_ = false;
  this->processes_per_dimension_[0] = 0;

  // the number of processes along the horizontal dimension should always be
  // equal to 1.
  this->processes_per_dimension_[1] = 1;
}

MPI_Comm Matrix::GetCartesianCommunicator() {
  return this->cartesian_communicator_;
}

Matrix Matrix::CloneShell() {
  Matrix m(this->halo_size_, this->matrix_width_, this->matrix_height_);
  m.neighbours = this->neighbours;
  m.partition_coords_[0] = this->partition_coords_[0];
  m.partition_coords_[1] = this->partition_coords_[1];
  m.processes_per_dimension_[0] = this->processes_per_dimension_[0];
  m.processes_per_dimension_[1] = this->processes_per_dimension_[1];
  m.partition_height_ = this->partition_height_;
  m.proc_id_ = this->proc_id_;
  m.proc_count_ = this->proc_count_;
  m.y_partitions_ = this->y_partitions_;
  m.is_initialized_ = this->is_initialized_;
  m.right_grid_border_coord_ = this->right_grid_border_coord_;
  m.left_grid_border_coord_ = this->left_grid_border_coord_;
  m.bottom_grid_border_coord_ = this->bottom_grid_border_coord_;
  m.top_grid_border_coord_ = this->top_grid_border_coord_;
  m.cartesian_communicator_ = this->cartesian_communicator_;
  m.ConfigGpuExecution();
  m.InitData(this->initial_inner_value_, this->initial_left_value_,
             this->initial_right_value_, this->initial_bottom_value_,
             this->initial_top_value_);
  return m;
}

void Matrix::InitData(float inner_value, float left_border_value,
                      float right_border_value, float bottom_border_value,
                      float top_border_value) {
  this->initial_inner_value_ = inner_value;
  this->initial_left_value_ = left_border_value;
  this->initial_right_value_ = right_border_value;
  this->initial_bottom_value_ = bottom_border_value;
  this->initial_top_value_ = top_border_value;
  int inner_data_width = this->matrix_width_;
  int inner_data_rows = this->partition_height_ +
                        2; // Two extra rows to hold the top and bottom halo
  this->AllocateMemory();
  int row = 0;
  int col;

  // Top halo initialization
  // If the partition holds the top border of the matrix, there is no halo to
  // init.
  float halo_value_inner = this->IsTopBorder() ? -1 : inner_value;
  float halo_value_left = this->IsTopBorder() ? -1 : left_border_value;
  float halo_value_right = this->IsTopBorder() ? -1 : right_border_value;
  this->inner_points_[0] = halo_value_left;

  for (col = 1; col < inner_data_width - 1; col++) {
    this->inner_points_[col] = halo_value_inner;
  }

  this->inner_points_[inner_data_width - 1] = halo_value_right;

  // Bottom Halo initialization
  // If the partition holds the bottom border of the matrix, there is no halo to
  // init.

  halo_value_inner = this->IsBottomBorder() ? -1 : inner_value;
  halo_value_left = this->IsBottomBorder() ? -1 : left_border_value;
  halo_value_right = this->IsBottomBorder() ? -1 : right_border_value;

  row = inner_data_rows - 1;

  // the value on the first column of the halo will come from the left border
  this->inner_points_[row * inner_data_width] = halo_value_left;

  for (col = 1; col < inner_data_width - 1; col++) {
    this->inner_points_[row * inner_data_width + col] = halo_value_inner;
  }
  // the value on the last column of the halo will come from the right border
  this->inner_points_[inner_data_width * inner_data_rows - 1] =
      halo_value_right;

  // Inner data
  for (int i = 1; i < inner_data_rows - 1; i++) {
    for (int j = 0; j < inner_data_width; j++) {
      if (this->IsTopBorder() && i == 1) {
        this->SetLocal(top_border_value, i, j);
      } else if (this->IsBottomBorder() && i == (inner_data_rows - 2)) {
        this->SetLocal(bottom_border_value, i, j);
      } else {
        this->SetLocal(inner_value, i, j);
      }
    }
  }

  // Move data to GPU
  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    // Assign the pointer to the copy of gpu data on the host. gpu_region_start
    // - 1 rows must be skipped.
    this->inner_data_execution_plans_[i].h_data = &(
        this->inner_points_
            [(this->inner_data_execution_plans_[i].gpu_region_start - 1) *
             (this->inner_data_execution_plans_[i].GetAbsoluteGpuDataWidth())]);
    cudaSetDevice(this->inner_data_execution_plans_[i].gpu_id);
    cudaMemcpyAsync(
        this->inner_data_execution_plans_[i].d_data,
        this->inner_data_execution_plans_[i].h_data,
        (this->inner_data_execution_plans_[i].GetAbsoluteGpuDataWidth()) *
            (this->inner_data_execution_plans_[i].GetAbsoluteGpuDataHeight()) *
            sizeof(float),
        cudaMemcpyHostToDevice, this->inner_data_execution_plans_[i].stream);
  }

  // for (int i = 0; i < 2; i++)
  // {
  //     if ((i == 0 && !this->IsTopBorder()) ||
  //         (i == 1 && !this->IsBottomBorder()))
  //     {
  //         switch (i)
  //         {
  //         case 0:
  //             /* code */
  //             cudaSetDevice(this->border_execution_plans_[i].gpu_id);
  //             cudaMemcpyAsync(this->border_execution_plans_[i].d_data,
  //                             this->inner_points_ +
  //                             this->border_execution_plans_[i].gpu_region_start,
  //                             (this->border_execution_plans_[i].gpu_data_width
  //                             + 2) * sizeof(float), cudaMemcpyHostToDevice,
  //                             this->border_execution_plans_[i].stream);
  //             cudaMemcpyAsync(this->border_execution_plans_[i].d_data +
  //             this->border_execution_plans_[i].gpu_data_width + 2,
  //                             this->inner_points_ + this->matrix_width_ +
  //                             this->border_execution_plans_[i].gpu_region_start,
  //                             (this->border_execution_plans_[i].gpu_data_width
  //                             + 2) * sizeof(float), cudaMemcpyHostToDevice,
  //                             this->border_execution_plans_[i].stream);

  //             cudaMemcpyAsync(this->border_execution_plans_[i].d_data +
  //             (this->border_execution_plans_[i].gpu_data_width + 2) * 2,
  //                             this->inner_points_ + 2 * this->matrix_width_ +
  //                             this->border_execution_plans_[i].gpu_region_start,
  //                             (this->border_execution_plans_[i].gpu_data_width
  //                             + 2) * sizeof(float), cudaMemcpyHostToDevice,
  //                             this->border_execution_plans_[i].stream);
  //             break;
  //         case 1:
  //             // Inner halo for the border
  //             int host_offset = (this->partition_height_ - 1) *
  //             this->matrix_width_; float *first_row_to_copy =
  //             this->inner_points_ + host_offset;
  //             cudaSetDevice(this->border_execution_plans_[i].gpu_id);
  //             cudaMemcpyAsync(this->border_execution_plans_[i].d_data,
  //                             first_row_to_copy +
  //                             this->border_execution_plans_[i].gpu_region_start,
  //                             (this->border_execution_plans_[i].gpu_data_width
  //                             + 2) * sizeof(float), cudaMemcpyHostToDevice,
  //                             this->border_execution_plans_[i].stream);
  //             cudaMemcpyAsync(this->border_execution_plans_[i].d_data +
  //             this->border_execution_plans_[i].gpu_data_width + 2,
  //                             first_row_to_copy + this->matrix_width_ +
  //                             this->border_execution_plans_[i].gpu_region_start,
  //                             (this->border_execution_plans_[i].gpu_data_width
  //                             + 2) * sizeof(float), cudaMemcpyHostToDevice,
  //                             this->border_execution_plans_[i].stream);

  //             cudaMemcpyAsync(this->border_execution_plans_[i].d_data +
  //             (this->border_execution_plans_[i].gpu_data_width + 2) * 2,
  //                             first_row_to_copy + 2 * this->matrix_width_ +
  //                             this->border_execution_plans_[i].gpu_region_start,
  //                             (this->border_execution_plans_[i].gpu_data_width
  //                             + 2) * sizeof(float), cudaMemcpyHostToDevice,
  //                             this->border_execution_plans_[i].stream);
  //             break;
  //         default:
  //             // TODO (endizhupani@uni-muenster.de): throw an exception
  //             break;
  //         }
  //     }
  // }
}

void Matrix::Init(float value, int argc, char *argv[]) {
  this->Init(value, value, value, value, value, argc, argv);
}

void Matrix::ConfigGpuExecution() {
  cudaDeviceProp device_prop;

  int current_device = 0;

  for (int i = 0; i < 2; i++) {

    CUDA_CHECK_RETURN(cudaSetDevice(current_device));
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&device_prop, current_device));
    border_execution_plans_[i].SetDeviceProperties(device_prop);
    border_execution_plans_[i].gpu_id = current_device;
    CUDA_CHECK_RETURN(cudaStreamCreate(&(border_execution_plans_[i].stream)));

    // The stream is created for a border that is NOT also a matrix border which
    // means that the calculation will be performed. In this case, if possible,
    // the device should be changed.
    if ((i == 0 && !this->IsTopBorder()) ||
        (i == 1 && !this->IsBottomBorder())) {
      current_device++;
      if (current_device == this->matrix_config_.gpu_number) {
        // Return back to the first device if we were on the last device.
        current_device = 0;
      }
    }
  }

  this->inner_data_execution_plans_ =
      new GpuExecution[this->matrix_config_.gpu_number];
  // configure inner data streams
  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    CUDA_CHECK_RETURN(cudaSetDevice(i));
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&device_prop, i));
    this->inner_data_execution_plans_[i].SetDeviceProperties(device_prop);
    CUDA_CHECK_RETURN(
        cudaStreamCreate(&(this->inner_data_execution_plans_[i].stream)));
    if (this->inner_data_execution_plans_[i].GetConcurrentKernelAndCopy()) {
      CUDA_CHECK_RETURN(cudaStreamCreate(
          &(this->inner_data_execution_plans_[i].auxilary_copy_stream)));
      CUDA_CHECK_RETURN(cudaStreamCreate(
          &(this->inner_data_execution_plans_[i].auxilary_calculation_stream)));
    }
    this->inner_data_execution_plans_[i].gpu_id = i;
  }
}

void Matrix::Deallocate() {
  cudaFreeHost(&inner_points_);
  for (int i = 0; i < 2; i++) {
    cudaFree(this->border_execution_plans_[i].d_data);
  }

  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    cudaFree(this->inner_data_execution_plans_[i].d_data);
  }
  // TODO (endizhupani@uni-muenster.de): Add code to deallocate the rest of the
  // data.
}

void Matrix::AllocateMemory() {
  int inner_data_rows, inner_rows, total_gpu_rows, rows_per_gpu,
      rows_to_allocate, rows_allocated, inner_rows_start_row_idx;

  inner_data_rows = this->partition_height_ +
                    2; // Two extra rows to hold the top and bottom halo
  inner_rows = this->partition_height_ - 2;
  inner_rows_start_row_idx = 2; // skip top halo and border.
  total_gpu_rows = this->matrix_config_.GpuRows(inner_rows);

  // Does not include halo rows.
  rows_per_gpu =
      floor(((float)total_gpu_rows) / this->matrix_config_.gpu_number);
  rows_allocated = 0;

  CUDA_CHECK_RETURN(
      cudaMallocHost(&(this->inner_points_),
                     inner_data_rows * this->matrix_width_ * sizeof(float)));

  // convenient reference to the top halo.
  this->top_halo_ = &this->inner_points_[0];

  // convenient reference to the bottom halo
  this->bottom_halo_ =
      &this->inner_points_[(inner_data_rows - 1) * this->matrix_width_];

  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    auto gpu_execution_plan = this->GetInnerDataPlanForGpuId(i);

    // last device should get the remainer of the rows.
    rows_to_allocate = (i = this->matrix_config_.gpu_number - 1)
                           ? (total_gpu_rows - rows_allocated)
                           : rows_per_gpu;
    // left and right border are excluded.
    gpu_execution_plan.SetGpuDataWidth(this->matrix_width_ - 2);
    gpu_execution_plan.SetGpuDataHeight(rows_per_gpu);
    cudaSetDevice(gpu_execution_plan.gpu_id);

    GpuReductionOperation op;
    op.gpu_id = i;
    float *d_difference_vector;
    CUDA_CHECK_RETURN(cudaMalloc(&d_difference_vector,
                                 gpu_execution_plan.GetGpuDataHeight() *
                                     gpu_execution_plan.GetGpuDataWidth() *
                                     sizeof(float)));
    op.d_vector_to_reduce = d_difference_vector;
    op.vector_to_reduce_length = gpu_execution_plan.GetGpuDataHeight() *
                                 gpu_execution_plan.GetGpuDataWidth();

    float *d_reduction_result;
    CUDA_CHECK_RETURN(cudaMalloc(&d_reduction_result, sizeof(float)));
    op.d_reduction_result = d_reduction_result;

    CUDA_CHECK_RETURN(cudaMalloc(
        &(gpu_execution_plan.d_data),
        gpu_execution_plan.GetAbsoluteGpuDataHeight() *
            gpu_execution_plan.GetAbsoluteGpuDataWidth() * sizeof(float)));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Does not perform any calculations. Simply returns the temp storage
    // requirements.
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           d_difference_vector, d_reduction_result,
                           gpu_execution_plan.GetGpuDataHeight() *
                               gpu_execution_plan.GetGpuDataWidth());

    CUDA_CHECK_RETURN(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    op.tmp_data_size_in_bytes = temp_storage_bytes;
    op.d_tmp_data = d_temp_storage;

    this->inner_data_reduction_plans_.push_back(op);

    gpu_execution_plan.gpu_region_start = inner_rows +
                                          inner_rows_start_row_idx - 1 -
                                          (total_gpu_rows - rows_allocated);
    rows_allocated += rows_per_gpu;
    gpu_execution_plan.is_contiguous_on_host = true;
  }

  // TODO (endizhupani@uni-muenster.de): Enable after the inner data calculation
  // starts working. for (int i = 0; i < 2; i++)
  // {
  //     CUDA_CHECK_RETURN(cudaSetDevice(border_execution_plans_[i].gpu_id));
  //     if (
  //         (i == 0 && !this->IsTopBorder()) ||
  //         (i == 1 && !this->IsBottomBorder()))
  //     {
  //         // the top and bottom border are two elements smaller then the
  //         partition width. int gpu_elements =
  //         CalculateGpuRows(this->matrix_width_ - 2, 1 -
  //         this->matrix_config_.cpu_perc); gpu_elements += 2; // add two for
  //         the left and right halo cells. gpu_elements *= 3; // multiply by 3
  //         because three columns will need to be stored. the top nd bottom
  //         halo of each border needs to be stored.
  //         CUDA_CHECK_RETURN(cudaMalloc(&(border_execution_plans_[i].d_data),
  //         gpu_elements * sizeof(float)));

  //         // remove one from the matrix width
  //         this->border_execution_plans_[i].gpu_region_start =
  //         this->matrix_width_
  //         - 1 - gpu_elements;
  //     }
  // }
}

void Matrix::Init(float inner_value, float left_border_value,
                  float right_border_value, float bottom_border_value,
                  float top_border_value, int argc, char *argv[]) {
  if (!is_initialized_) {
    this->InitializeMPI(argc, argv);
    this->is_initialized_ = true;
  }

  if (this->matrix_height_ % this->processes_per_dimension_[0] != 0) {
    throw new std::logic_error(
        "The height of the matrix must be divisable by the number of processes "
        "along the y dimension: " +
        std::to_string(this->processes_per_dimension_[0]));
  }

  // This is the partition height without the ghost points.
  this->partition_height_ =
      this->matrix_height_ / this->processes_per_dimension_[0];
  this->ConfigGpuExecution();
  this->InitData(inner_value, left_border_value, right_border_value,
                 bottom_border_value, top_border_value);
  MPI_Barrier(this->GetCartesianCommunicator());
}

void Matrix::PrintMatrixInfo() {
  if (this->proc_id_ == 0) {
    printf("Number of processes: %d\n", this->proc_count_);
    // int deviceCount = 0;
    // printf("Number of GPUs: %d\n", deviceCount);
  }
  MPI_Barrier(this->GetCartesianCommunicator());
  std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: ("
            << this->partition_coords_[0] << ", " << this->partition_coords_[1]
            << "); Top ID: " << this->neighbours[0].GetNeighborId()
            << "; Bottom ID: " << this->neighbours[1].GetNeighborId()
            << "; Partition size: " << this->matrix_width_ << "x"
            << this->partition_height_ << std::endl;

  MPI_Barrier(this->GetCartesianCommunicator());
}

void Matrix::RankByCoordinates(const int coords[2], int *rank) {
  if (coords[0] < 0 || coords[1] < 0 ||
      coords[0] >= processes_per_dimension_[0] ||
      coords[1] >= processes_per_dimension_[1]) {
    *rank = MPI_PROC_NULL;
    return;
  }
  MPI_Cart_rank(this->GetCartesianCommunicator(), coords, rank);
}

void Matrix::InitializeMPI(int argc, char *argv[]) {
  if (is_initialized_) {
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

  MPI_Cart_create(MPI_COMM_WORLD, n_dim, this->processes_per_dimension_,
                  periods, 1, &this->cartesian_communicator_);
  MPI_Comm_rank(this->GetCartesianCommunicator(), &proc_id_);
  // Fetch the coordinates of the current partition
  MPI_Cart_coords(this->GetCartesianCommunicator(), this->proc_id_, 2,
                  this->partition_coords_);

  this->neighbours[0].type = PartitionNeighbourType::TOP_NEIGHBOUR;
  int neighbor_coords[2] = {this->partition_coords_[0] - 1,
                            this->partition_coords_[1]};
  this->RankByCoordinates(neighbor_coords, &this->neighbours[0].id);

  this->neighbours[1].type = PartitionNeighbourType::BOTTOM_NEIGHBOUR;
  neighbor_coords[0] = this->partition_coords_[0] + 1;
  neighbor_coords[1] = this->partition_coords_[1];
  this->RankByCoordinates(neighbor_coords, &this->neighbours[1].id);
}

void Matrix::Finalize() {
  Deallocate();
  MPI_Barrier(this->GetCartesianCommunicator());
  int result = MPI_Finalize();
  if (this->proc_id_ == 0) {
    printf("MPi Finalize, process %d, return code: %d\n", this->proc_id_,
           result);
  }
}

const GpuExecution &Matrix::GetCpuAdjacentInnerDataGpuPlan() {
  return this->GetInnerDataPlanForGpuId(0);
}

const GpuExecution &Matrix::GetInnerDataPlanForGpuId(int gpu_id) {
  for (size_t i = 0; i < this->matrix_config_.gpu_number; i++) {
    if (this->inner_data_execution_plans_[i].gpu_id == gpu_id) {
      return this->inner_data_execution_plans_[i];
    }
  }
}

const GpuExecution Matrix::GetRightBorderStream() {
  return this->border_execution_plans_[1];
}

const GpuExecution Matrix::GetLeftBorderStream() {
  return this->border_execution_plans_[0];
}

float Matrix::LocalSweep(Matrix new_matrix, ExecutionStats *execution_stats) {
  // for (int i = 0; i < this->matrix_config_.gpu_number; i++)
  // {
  //     auto stream = this->GetInnerDataPlanForGpuId(i);
  //     cudaSetDevice(stream.gpu_id);

  //     // move top border to the host data. Top halo of the GPU data will be
  //     skipped. Left and right halo must be skipped.
  //     cudaMemcpyAsync(stream.h_data + stream.GetAbsoluteGpuDataWidth() + 1,
  //                     stream.d_data + stream.GetAbsoluteGpuDataWidth() + 1,
  //                     stream.GetGpuDataWidth() * sizeof(float),
  //                     cudaMemcpyDeviceToHost,
  //                     stream.stream);

  //     // move bottom border to the host data. Left and right halo must be
  //     skipped. cudaMemcpyAsync(stream.h_data +
  //     (stream.GetAbsoluteGpuDataWidth() * stream.GetGpuDataHeight()) + 1,
  //                     stream.d_data + (stream.GetAbsoluteGpuDataWidth() *
  //                     stream.GetGpuDataHeight()) + 1,
  //                     stream.GetGpuDataWidth() * sizeof(float),
  //                     cudaMemcpyDeviceToHost,
  //                     stream.stream);
  // }

  // for (int i = 0; i < this->matrix_config_.gpu_number; i++)
  // {
  //     auto stream = this->GetInnerDataPlanForGpuId(i);
  //     cudaSetDevice(stream.gpu_id);
  //     cudaStreamSynchronize(stream.stream);
  // }

  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    auto stream = this->GetInnerDataPlanForGpuId(i);
    cudaSetDevice(stream.gpu_id);

    // move top halo to the gpu data
    cudaMemcpyAsync(stream.d_data + 1, stream.h_data + 1,
                    stream.GetGpuDataWidth() * sizeof(float),
                    cudaMemcpyHostToDevice, stream.stream);

    // move bottom halo to the gpu data
    cudaMemcpyAsync(
        stream.d_data +
            stream.GetAbsoluteGpuDataWidth() * (stream.GetGpuDataHeight() + 1) +
            1,
        stream.h_data +
            stream.GetAbsoluteGpuDataWidth() * (stream.GetGpuDataHeight() + 1) +
            1,
        stream.GetGpuDataWidth() * sizeof(float), cudaMemcpyHostToDevice,
        stream.stream);
  }

  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    auto stream = this->GetInnerDataPlanForGpuId(i);
    cudaSetDevice(stream.gpu_id);
    cudaStreamSynchronize(stream.stream);
  }

  this->current_max_difference_ = 0;
  float diff, new_value;

  int max_thread_num = omp_get_max_threads();

  // 2 sends and 2 receives
  MPI_Request requests[4];

  // Two levels of parallelization
  omp_set_nested(2);
  auto sweep_start = MPI_Wtime();
  auto border_start = MPI_Wtime();
#pragma omp parallel sections firstprivate(diff, new_value, max_thread_num)    \
    num_threads(std::min(max_thread_num, 2))                                   \
        reduction(max                                                          \
                  : current_max_difference_)
  {

#pragma omp section
    {
      if (!this->IsTopBorder()) {
#pragma omp parallel for reduction(max : current_max_difference_)
        for (int i = 1; i < this->matrix_width_ - 1; i++) {
          new_value = (this->GetLocal(0, i - 1) + this->GetLocal(0, i + 1) +
                       this->GetLocal(1, i) + this->top_halo_[i]) /
                      4;
          diff = fabs(new_value - this->GetLocal(0, i));
          if (diff > current_max_difference_) {
            current_max_difference_ = diff;
          }
          new_matrix.SetLocal(new_value, 0, i);
        }

        auto neighbour_top =
            this->GetNeighbour(PartitionNeighbourType::TOP_NEIGHBOUR);
        MPI_Isend(&new_matrix.inner_points_[this->matrix_width_],
                  this->matrix_width_, MPI_DOUBLE, neighbour_top.id, 0,
                  this->GetCartesianCommunicator(), &requests[0]);
        MPI_Irecv(new_matrix.top_halo_, this->matrix_width_, MPI_DOUBLE,
                  neighbour_top.id, MPI_ANY_TAG,
                  this->GetCartesianCommunicator(), &requests[2]);
      } else {
        MPI_Isend(&new_matrix.inner_points_[this->matrix_width_],
                  this->matrix_width_, MPI_DOUBLE, MPI_PROC_NULL, 0,
                  this->GetCartesianCommunicator(), &requests[0]);

        MPI_Irecv(new_matrix.top_halo_, this->matrix_width_, MPI_DOUBLE,
                  MPI_PROC_NULL, MPI_ANY_TAG, this->GetCartesianCommunicator(),
                  &requests[2]);
      }
    }

#pragma omp section
    {
      if (!this->IsBottomBorder()) {
        int bottom_border_row = this->partition_height_ - 1;
#pragma omp parallel for firstprivate(bottom_border_row)                       \
    reduction(max                                                              \
              : current_max_difference_)
        for (int i = 1; i < this->matrix_width_ - 1; i++) {
          new_value = (this->GetLocal(bottom_border_row, i - 1) +
                       this->GetLocal(bottom_border_row, i + 1) +
                       this->GetLocal(bottom_border_row - 1, i) +
                       this->bottom_halo_[i]) /
                      4;
          if (diff > current_max_difference_) {
            current_max_difference_ = diff;
          }
          new_matrix.SetLocal(new_value, bottom_border_row, i);
        }

        auto neighbour_bottom =
            this->GetNeighbour(PartitionNeighbourType::BOTTOM_NEIGHBOUR);
        MPI_Isend(&new_matrix.inner_points_[this->partition_height_ *
                                            this->matrix_width_],
                  this->matrix_width_, MPI_DOUBLE, neighbour_bottom.id, 0,
                  this->GetCartesianCommunicator(), &requests[1]);
        MPI_Irecv(new_matrix.bottom_halo_, this->matrix_width_, MPI_DOUBLE,
                  neighbour_bottom.id, MPI_ANY_TAG,
                  this->GetCartesianCommunicator(), &requests[3]);
      } else {
        MPI_Isend(&new_matrix.inner_points_[this->partition_height_ *
                                            this->matrix_width_],
                  this->matrix_width_, MPI_DOUBLE, MPI_PROC_NULL, 0,
                  this->GetCartesianCommunicator(), &requests[1]);
        MPI_Irecv(new_matrix.bottom_halo_, this->matrix_width_, MPI_DOUBLE,
                  MPI_PROC_NULL, MPI_ANY_TAG, this->GetCartesianCommunicator(),
                  &requests[3]);
      }
    }
  }
  execution_stats->total_border_calc_time += (MPI_Wtime() - border_start);

  auto inner_points_time = MPI_Wtime();
  float *max_differences = new float[this->matrix_config_.gpu_number];
  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    GpuExecution execution_plan = this->GetInnerDataPlanForGpuId(i);
    if (execution_plan.GetConcurrentKernelAndCopy()) {
      this->ExecuteGpuWithConcurrentCopy(new_matrix, execution_plan);
    }
  }

  auto cpu_adjacent_stream = this->GetCpuAdjacentInnerDataGpuPlan();
#pragma omp parallel reduction(max : current_max_difference_)
  {
    int i, j;
#pragma omp for collapse(2) private(i, j, diff, new_value)
    for (i = 1; i < cpu_adjacent_stream.gpu_region_start; i++)
      for (j = 1; j < this->matrix_width_ - 1; j++) {
        new_value = (this->GetLocal(i - 1, j) + this->GetLocal(i + 1, j) +
                     this->GetLocal(i, j - 1) + this->GetLocal(i, j + 1)) /
                    4;
        diff = fabs(new_value - this->GetLocal(i, j));
        if (diff > current_max_difference_) {
          current_max_difference_ = diff;
        }
        new_matrix.SetLocal(new_value, i, j);
      }
  }

  execution_stats->total_inner_points_time += (MPI_Wtime() - inner_points_time);

  auto idle_start = MPI_Wtime();
  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
  for (int i = 0; i < this->matrix_config_.gpu_number; i++) {
    GpuExecution execution_plan = this->GetInnerDataPlanForGpuId(i);
    CUDA_CHECK_RETURN(cudaSetDevice(execution_plan.gpu_id));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(execution_plan.stream));

    if (execution_plan.GetConcurrentKernelAndCopy()) {
      CUDA_CHECK_RETURN(
          cudaStreamSynchronize(execution_plan.auxilary_calculation_stream));
      CUDA_CHECK_RETURN(
          cudaStreamSynchronize(execution_plan.auxilary_copy_stream));
    }
  }

  execution_stats->total_idle_comm_time += (MPI_Wtime() - idle_start);
  execution_stats->total_sweep_time += (MPI_Wtime() - sweep_start);
  execution_stats->n_sweeps += 1;
  return this->current_max_difference_;
}

const GpuReductionOperation &Matrix::GetInnerReductionOperation(int gpu_id) {
  for (size_t i = 0; i < this->inner_data_reduction_plans_.size(); i++) {
    if (this->inner_data_reduction_plans_[i].gpu_id == gpu_id) {
      return this->inner_data_reduction_plans_[i];
    }
  }
}

void Matrix::ExecuteGpuWithConcurrentCopy(Matrix new_matrix,
                                          GpuExecution gpu_execution_plan) {
  auto reduction_operation =
      this->GetInnerReductionOperation(gpu_execution_plan.gpu_id);
  GpuExecution new_matrix_stream =
      new_matrix.GetInnerDataPlanForGpuId(gpu_execution_plan.gpu_id);

  dim3 block_size(gpu_execution_plan.GetGpuBlockSizeX(),
                  gpu_execution_plan.GetGpuBlockSizeY());
  dim3 grid_size(gpu_execution_plan.GetGpuGridSizeX(),
                 gpu_execution_plan.GetGpuBlockSizeY());

  CUDA_CHECK_RETURN(cudaSetDevice(gpu_execution_plan.gpu_id));

  size_t shared_mem_size = gpu_execution_plan.GetAbsoluteGpuDataHeight() *
                           gpu_execution_plan.GetAbsoluteGpuDataWidth() *
                           sizeof(float);

  // when this event is recorded, the difference calculation and the border
  // transfers can start.
  cudaEvent_t kernel_computation_complete;
  CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&kernel_computation_complete,
                                             cudaEventDisableTiming));

  jacobiKernel<<<grid_size, block_size, shared_mem_size,
                 gpu_execution_plan.stream>>>(
      gpu_execution_plan.d_data, new_matrix_stream.d_data,
      reduction_operation.d_vector_to_reduce,
      gpu_execution_plan.GetGpuDataWidth(),
      gpu_execution_plan.GetGpuDataHeight());
  CUDA_CHECK_RETURN(
      cudaEventRecord(kernel_computation_complete, gpu_execution_plan.stream));
  CUDA_CHECK_RETURN(
      cudaStreamWaitEvent(gpu_execution_plan.auxilary_calculation_stream,
                          kernel_computation_complete, 0));

  cub::DeviceReduce::Max(reduction_operation.d_tmp_data,
                         reduction_operation.tmp_data_size_in_bytes,
                         reduction_operation.d_vector_to_reduce,
                         reduction_operation.d_reduction_result,
                         reduction_operation.vector_to_reduce_length,
                         gpu_execution_plan.auxilary_calculation_stream);

  int transfer_start_idx = gpu_execution_plan.GetAbsoluteGpuDataWidth() + 1;

  // move top border to the host data. Top halo of the GPU data will be skipped.
  // Left and right halo must be skipped.
  CUDA_CHECK_RETURN(
      cudaMemcpyAsync(gpu_execution_plan.h_data + transfer_start_idx,
                      gpu_execution_plan.d_data + transfer_start_idx,
                      gpu_execution_plan.GetGpuDataWidth() * sizeof(float),
                      cudaMemcpyDeviceToHost, gpu_execution_plan.stream));

  // move bottom border to the host data. Left and right halo must be skipped.
  CUDA_CHECK_RETURN(cudaStreamWaitEvent(gpu_execution_plan.auxilary_copy_stream,
                                        kernel_computation_complete, 0));
  transfer_start_idx = (gpu_execution_plan.GetAbsoluteGpuDataWidth() *
                        gpu_execution_plan.GetGpuDataHeight()) +
                       1;
  CUDA_CHECK_RETURN(cudaMemcpyAsync(
      gpu_execution_plan.h_data + transfer_start_idx,
      gpu_execution_plan.d_data + transfer_start_idx,
      gpu_execution_plan.GetGpuDataWidth() * sizeof(float),
      cudaMemcpyDeviceToHost, gpu_execution_plan.auxilary_copy_stream));
}

const PartitionNeighbour
Matrix::GetNeighbour(PartitionNeighbourType neighbour_type) {
  switch (neighbour_type) {
  case PartitionNeighbourType::TOP_NEIGHBOUR:
    return this->neighbours[0];
  case PartitionNeighbourType::BOTTOM_NEIGHBOUR:
    return this->neighbours[1];
  default:
    throw new std::out_of_range(
        "The neighbour type must be TOP_NEIGHBOUR, BOTTOM_NEIGHBOUR, "
        "RIGHT_NEIGHBOUR or LEFT_NEIGHBOUR");
  }
}

const float Matrix::GetLocal(int partition_row, int partition_col) {
  if (partition_col < 0 || partition_col > this->matrix_width_ - 1 ||
      partition_row < 0 || partition_row > this->partition_height_ - 1) {
    throw new std::out_of_range("Index is out of range for the partition");
  }

  int row_offset = (partition_row + 1) * this->matrix_width_;
  return this->inner_points_[row_offset + partition_col];
}

void Matrix::SetLocal(float value, int row, int col) {
  // offset the row by one because the first row of inner data will hold the top
  // halo.
  this->inner_points_[(row + 1) * this->matrix_width_ + col] = value;
}

void Matrix::SetGlobal(float value, int row, int col) {
  if (row < 0 || col < 0) {
    return;
  }

  if (row > this->matrix_height_ - 1) {
    return;
  }

  if (col > this->matrix_width_ - 1) {
    return;
  }

  int partition_row_start =
      this->partition_coords_[0] * this->partition_height_;
  int partition_row_end = this->partition_coords_[0] * this->partition_height_ +
                          this->partition_height_ - 1;

  if (row < partition_row_start || row > partition_row_end) {
    return;
  }

  int local_row = row - partition_row_start;
  this->SetLocal(value, local_row, col);
}

const float Matrix::GlobalDifference(ExecutionStats *execution_stats) {
  MPI_Barrier(this->GetCartesianCommunicator());
  auto reduction_start = MPI_Wtime();
  float received_difference = 0;
  MPI_Allreduce(&current_max_difference_, &received_difference, 1, MPI_DOUBLE,
                MPI_MAX, this->GetCartesianCommunicator());
  current_max_difference_ = received_difference;
  execution_stats->total_time_reducing_difference +=
      (MPI_Wtime() - reduction_start);
  execution_stats->n_diff_reducions += 1;
  return current_max_difference_;
}

void Matrix::ShowMatrix() {
  float *partition_data = inner_points_; // this->AssemblePartition();
  float *matrix;
  if (this->proc_id_ == 0) {
    matrix = new float[this->matrix_height_ * this->matrix_width_];
  }

  // Need to skip the first row because that is the top halo.
  MPI_Gather(&(partition_data[this->matrix_width_]),
             this->partition_height_ * this->matrix_width_, MPI_DOUBLE, matrix,
             this->partition_height_ * this->matrix_width_, MPI_DOUBLE, 0,
             this->GetCartesianCommunicator());
  for (int i = 0; i < this->matrix_height_; i++) {
    for (int j = 0; j < this->matrix_width_; j++) {
      printf("%6.2f ", matrix[i * this->matrix_width_ + j]);
    }
    putchar('\n');
  }

  delete[] matrix;
}

void Matrix::Synchronize() { MPI_Barrier(this->GetCartesianCommunicator()); }

void Matrix::GetPartitionCoordinates(int rank, int *partition_coords) {

  MPI_Cart_coords(this->GetCartesianCommunicator(), rank, 2, partition_coords);
}
void Matrix::PrintPartitionData() {
  std::cout << "Processor id: " << this->proc_id_ << "; Coordinates: ("
            << this->partition_coords_[0] << ", " << this->partition_coords_[1]
            << "); Top ID: " << this->neighbours[0].GetNeighborId()
            << "; Bottom ID: " << this->neighbours[1].GetNeighborId()
            << "; Partition size: " << this->partition_height_ << "x"
            << this->matrix_width_ << std::endl;

  for (int i = 0; i < this->partition_height_ + 2; i++) {

    int offset = i * (this->matrix_width_);
    for (int j = 0; j < this->matrix_width_; j++) {
      printf("%6.2f ", this->inner_points_[offset + j]);
    }
    printf("\n");
  }
}

void Matrix::PrintAllPartitions() {
  for (int i = 0; i < this->proc_count_; i++) {
    MPI_Barrier(this->GetCartesianCommunicator());
    if (i == this->proc_id_) {
      this->PrintPartitionData();
      fflush(stdout);
    }
  }
}

bool Matrix::IsTopBorder() {
  return this->partition_coords_[0] == this->top_grid_border_coord_;
}
bool Matrix::IsBottomBorder() {
  return this->partition_coords_[0] == this->bottom_grid_border_coord_;
}
} // namespace pde_solver
