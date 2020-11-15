#include "kernels.cuh"

#include <cub/device/device_reduce.cuh>
//#include <cub/util_allocator.cuh>

__global__ void jacobiKernel(float *in, float *out, float *diff,
                             int gpu_data_width, int gpu_data_height) {
  extern __shared__ float shared[];
  int absolute_gpu_data_width = gpu_data_width + 2;
  int absolute_gpu_data_height = gpu_data_height + 2;
  size_t y =
      blockIdx.y * blockDim.y + threadIdx.y + 1; // add one for the halo point
  size_t x =
      blockIdx.x * blockDim.x + threadIdx.x + 1; // add one for the halo point

  size_t local_y = threadIdx.y + 1;
  size_t local_x = threadIdx.x + 1;

  size_t smem_width = blockDim.x + 2;
  size_t smem_height = blockDim.y + 2;

  if (y >= gpu_data_height || x >= gpu_data_width) {
    return;
  }
  shared[local_y * (smem_width) + local_x] =
      in[y * absolute_gpu_data_width + x];

  // points on the borders need to also load the halo points into shared memory.
  if (local_y == 1) {
    shared[(local_y - 1) * (smem_width) + local_x] =
        in[(y - 1) * absolute_gpu_data_width + x];
  } else if (local_y == (smem_height - 2)) {
    shared[(local_y + 1) * (smem_width) + local_x] =
        in[(y + 1) * absolute_gpu_data_width + x];
  }

  if (local_x == 1) {
    shared[(local_y) * (smem_width) + local_x - 1] =
        in[y * absolute_gpu_data_width + x - 1];
  } else if (local_x == (smem_width - 2)) {
    shared[local_y * (smem_width) + local_x + 1] =
        in[y * absolute_gpu_data_width + x - 1];
  }

  __syncwarp();

  float val = (shared[(local_y - 1) * (smem_width) + local_x] +
               shared[(local_y + 1) * (smem_width) + local_x] +
               shared[(local_y) * (smem_width) + local_x + 1] +
               shared[(local_y) * (smem_width) + local_x - 1]) /
              4;

  float current_diff = shared[(local_y) * (smem_width) + local_x] - val;
  if (current_diff < 0)
    current_diff = current_diff * (-1);
  diff[y * absolute_gpu_data_width + x] = current_diff;
  out[y * absolute_gpu_data_width + x] =
      (shared[(local_y - 1) * (smem_width) + local_x] +
       shared[(local_y + 1) * (smem_width) + local_x] +
       shared[(local_y) * (smem_width) + local_x + 1] +
       shared[(local_y) * (smem_width) + local_x - 1]) /
      4;
}

void LaunchJacobiKernel(float *in, float *out, float *diff, int gpu_data_width,
                        int gpu_data_height, dim3 block_size, dim3 grid_size,
                        size_t shared_mem_size, cudaStream_t stream) {
  jacobiKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
      in, out, diff, gpu_data_width, gpu_data_height);
}

cudaError_t LaunchReductionOperation(GpuReductionOperation reduction_operation,
                                     cudaStream_t stream) {
  return cub::DeviceReduce::Max(reduction_operation.d_tmp_data,
                                reduction_operation.tmp_data_size_in_bytes,
                                reduction_operation.d_vector_to_reduce,
                                reduction_operation.d_reduction_result,
                                reduction_operation.vector_to_reduce_length,
                                stream);
}

cudaError_t PrepReductionOperation(GpuReductionOperation reduction_operation) {
  return cub::DeviceReduce::Max(reduction_operation.d_tmp_data,
                                reduction_operation.tmp_data_size_in_bytes,
                                reduction_operation.d_vector_to_reduce,
                                reduction_operation.d_reduction_result,
                                reduction_operation.vector_to_reduce_length);
}

// TODO: Look into using thrust for the reduction operation.