// kernels.cuh
//
// ------------------------------------------------------
// Copyright (c) 2020 Endi Zhupani
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute,  sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef KERNELS
#define KERNELS
#include "common.h"
__global__ void jacobiKernel(float *in, float *out, float *diff,
                             int gpu_data_width, int gpu_data_height,
                             int smem_width, int smem_height);

void LaunchJacobiKernel(float *in, float *out, float *diff, int gpu_data_width,
                        int gpu_data_height, int smem_width, int smem_height,
                        dim3 block_size, dim3 grid_size, size_t shared_mem_size,
                        cudaStream_t stream);

cudaError_t LaunchReductionOperation(GpuReductionOperation &reduction_operation,
                                     cudaStream_t stream);

cudaError_t PrepReductionOperation(GpuReductionOperation &reduction_operation);

#endif