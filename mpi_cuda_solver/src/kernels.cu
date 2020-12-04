#include "kernels.cuh"

#include <cub/cub.cuh>
//#include <cub/util_allocator.cuh>

__global__ void jacobiKernel(float *in, float *out, float *diff,
                             int gpu_data_width, int gpu_data_height,
                             int smem_width, int smem_height,
                             int gpu_calc_start_r_idx, bool is_top_process,
                             bool is_bottom_process, int partition_height) {
  extern __shared__ float shared[];
  int glob_ty = blockIdx.y * blockDim.y + threadIdx.y;
  int glob_tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (glob_ty >= gpu_data_height || glob_tx >= gpu_data_width) {
    return;
  }

  int absolute_gpu_data_width = gpu_data_width;
  int absolute_gpu_data_height = gpu_data_height + 2;
  int global_memory_y = glob_ty + 1; // add one for the halo point
  int global_memory_x = glob_tx;     // add one for the halo point
  int partition_row = glob_ty + gpu_calc_start_r_idx;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int shared_memory_y = ty + 1;
  int shared_memory_x = tx + 1;

  shared[shared_memory_y * (smem_width) + shared_memory_x] =
      in[global_memory_y * absolute_gpu_data_width + global_memory_x];

  // points on the borders need to also load the halo points into shared memory.
  if (tx == 0) {
    if (glob_tx == 0) {
      shared[shared_memory_y * (smem_width)] = 100;
    } else {
      shared[(shared_memory_y) * (smem_width)] =
          in[global_memory_y * absolute_gpu_data_width + global_memory_x - 1];
    }
  }

  // right col
  if (tx == blockDim.x - 1 || glob_tx == (gpu_data_width - 1)) {
    // global right column
    if (glob_tx == (gpu_data_width - 1)) {
      shared[shared_memory_y * (smem_width) + shared_memory_x + 1] = 100;
    } else {
      shared[(shared_memory_y) * (smem_width) + shared_memory_x + 1] =
          in[global_memory_y * absolute_gpu_data_width + global_memory_x + 1];
    }
  }

  if (ty == 0) {
    if (partition_row == 0 && is_top_process) {
      shared[(shared_memory_y - 1) * (smem_width) + shared_memory_x] = 100;
    } else {
      shared[(shared_memory_y - 1) * (smem_width) + shared_memory_x] =
          in[(global_memory_y - 1) * absolute_gpu_data_width + global_memory_x];
    }
  }

  if (ty == blockDim.y - 1 || glob_ty == (gpu_data_height - 1)) {
    if (partition_row == (partition_height - 1) && is_bottom_process) {
      shared[(shared_memory_y + 1) * (smem_width) + shared_memory_x] = 0;
    } else {
      shared[(shared_memory_y + 1) * (smem_width) + shared_memory_x] =
          in[(global_memory_y + 1) * absolute_gpu_data_width + global_memory_x];
    }
  }

  // __syncthreads();
  // if (glob_tx == 0 && glob_ty == 0) {
  //   printf("partition row %d \n", partition_row);
  //   for (int i = 0; i < smem_width; i++) {
  //     for (int j = 0; j < smem_height; j++) {
  //       char *format = "%6.2f ";

  //       if (j == smem_height - 1) {
  //         format = "%6.2f \n";
  //       }
  //       printf(format, shared[i * smem_width + j]);
  //     }
  //   }
  //   printf("\n");
  // }
  __syncthreads();

  float val = (shared[(shared_memory_y - 1) * (smem_width) + shared_memory_x] +
               shared[(shared_memory_y + 1) * (smem_width) + shared_memory_x] +
               shared[(shared_memory_y) * (smem_width) + shared_memory_x + 1] +
               shared[(shared_memory_y) * (smem_width) + shared_memory_x - 1]) /
              4;

  float current_diff =
      shared[(shared_memory_y) * (smem_width) + shared_memory_x] - val;

  if (current_diff < 0)
    current_diff *= (-1);
  // printf("\n\nLocal Thread idx: x:%d y:%d\nGlobal Thread Idx: x:%d "
  //        "y:%d\nCalculating point (%d,%d)\nShared memory number items: %d\n "
  //        "Value is:%f\nCurrentDifference is: %f\nValues used are: %f,
  //        %f,%f,%f", threadIdx.x, threadIdx.y, glob_tx, glob_ty,
  //        global_memory_x, global_memory_y, smem_width * smem_height, val,
  //        current_diff, shared[(shared_memory_y - 1) * (smem_width) +
  //        shared_memory_x], shared[(shared_memory_y + 1) * (smem_width) +
  //        shared_memory_x], shared[(shared_memory_y) * (smem_width) +
  //        shared_memory_x + 1], shared[(shared_memory_y) * (smem_width) +
  //        shared_memory_x - 1]);

  diff[glob_ty * gpu_data_width + glob_tx] = current_diff;
  out[global_memory_y * absolute_gpu_data_width + global_memory_x] = val;
  __syncthreads();
}

void LaunchJacobiKernel(float *in, float *out, float *diff, int gpu_data_width,
                        int gpu_data_height, int smem_width, int smem_height,
                        int gpu_calc_start_r_idx, bool is_top_process,
                        bool is_bottom_process, int partition_height,
                        dim3 block_size, dim3 grid_size, size_t shared_mem_size,
                        cudaStream_t stream) {
  jacobiKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
      in, out, diff, gpu_data_width, gpu_data_height, smem_width, smem_height,
      gpu_calc_start_r_idx, is_top_process, is_bottom_process,
      partition_height);
}

cudaError_t LaunchReductionOperation(GpuReductionOperation &reduction_operation,
                                     cudaStream_t stream) {
  return cub::DeviceReduce::Max(reduction_operation.d_tmp_data,
                                reduction_operation.tmp_data_size_in_bytes,
                                reduction_operation.d_vector_to_reduce,
                                reduction_operation.d_reduction_result,
                                reduction_operation.vector_to_reduce_length,
                                stream);
}

cudaError_t PrepReductionOperation(GpuReductionOperation &reduction_operation) {
  return cub::DeviceReduce::Max(reduction_operation.d_tmp_data,
                                reduction_operation.tmp_data_size_in_bytes,
                                reduction_operation.d_vector_to_reduce,
                                reduction_operation.d_reduction_result,
                                reduction_operation.vector_to_reduce_length);
}

// TODO: Look into using thrust for the reduction operation.