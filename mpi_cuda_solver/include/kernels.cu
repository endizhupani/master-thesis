#include "common.h"
#include "matrix.h"
#include <cooperative_groups.h>
__global__ void jacobiKernel(float *in, float *out, GPUStream stream) {
  extern __shared__ float shared[];
  int absolute_gpu_data_width = stream.gpu_data_width + 2;
  int absolute_gpu_data_height = stream.gpu_data_height + 2;
  size_t y =
      blockIdx.y * blockDim.y + threadIdx.y + 1; // add one for the halo point
  size_t x =
      blockIdx.x * blockDim.x + threadIdx.x + 1; // add one for the halo point

  size_t local_y = threadIdx.y + 1;
  size_t local_x = threadIdx.x + 1;

  size_t smem_width = blockDim.x + 2;
  size_t smem_height = blockDim.y + 2;

  if (y >= stream.gpu_data_height || x >= stream.gpu_data_width) {
    __syncthreads();
    return;
  }
  shared[local_y * (smem_width) + local_x] =
      in[y * absolute_gpu_data_width + x];

  // points on the borders need to also load the halo points into shared memory.
  if (local_y == 1) {
    shared[(local_y - 1) * (smem_width) + local_x] =
        in[(y - 1) * absolute_gpu_data_width + x];
  } else if (local_y == (smem_height_ - 2)) {
    shared[(local_y + 1) * (smem_width) + local_x] =
        in[(y + 1) * absolute_gpu_data_width + x]
  }

  if (local_x == 1) {
    shared[(local_y) * (smem_width) + local_x - 1] =
        in[y * absolute_gpu_data_width + x - 1];
  } else if (local_x == (smem_width_ - 2)) {
    shared[local_y * (smem_width) + local_x + 1] =
        in[y * absolute_gpu_data_width + x - 1]
  }

  __syncthreads();

  out[y * absolute_gpu_data_width + x] =
      (shared[(local_y - 1) * (smem_width) + local_x] +
       shared[(local_y + 1) * (smem_width) + local_x] +
       shared[(local_y) * (smem_width) + local_x + 1] +
       shared[(local_y) * (smem_width) + local_x - 1]) /
      4;

  __syncthreads();
}