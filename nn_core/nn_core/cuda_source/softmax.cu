#include "softmax.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


template <int threads>
__device__ float __sm_sum(
	float* a
) {
#pragma unroll
	for (uint i = threads / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) a[threadIdx.x] += a[threadIdx.x + i];
		
		__syncthreads();
	}

	return a[0];
}

__global__ void __softmax_1d(
	const float* a,
	float* b,
	cuint offset,
	cuint size,
	cuint iter
) {
	/*
	threads 1024
	blocks  iter, 
	*/

	extern __shared__ float sm[];
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	
	
}


void softmax(
	cudaStream_t* s,
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape,
	std::vector<uint>& axis
) {
	nn_type* work_space = NULL;

	if (axis.size() > 1) {
		uint least_ch = 0;
		size_t size = 1;

		for (cint n : in_shape) size *= n;
		check_cuda(cudaMalloc(&work_space, sizeof(nn_type) * size));
		check_cuda(cudaMemcpy(work_space, input, sizeof(nn_type) * size, cudaMemcpyDeviceToDevice));

		for (uint n = 0; n < in_shape[0]; ++n) {
			for (cuint c : axis) {

			}
		}
	}
}