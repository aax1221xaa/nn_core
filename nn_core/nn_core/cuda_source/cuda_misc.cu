#include "cuda_misc.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __transpose(
	const float* input,
	float* output,
	cuint n,
	cuint c,
	cuint h,
	cuint w
) {
	cuint tidx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint k_idx = tidx % (w * h);
	cuint k_count = tidx / (w * h);

	cuint row = k_count % c;
	cuint col = k_count / c;

	float* p_out = output + (row * (w * h * n) + col * (w * h));

	if (tidx < (n * h * w * c)) {
		p_out[k_idx] = input[tidx];
	}
}


void transpose(const tensor4d& input_size, const nn_type* input, nn_type* output) {
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, input_size._n * input_size._c * input_size._h * input_size._w);

	__transpose<<<blocks, threads>>>(
		input,
		output,
		input_size._n,
		input_size._c,
		input_size._h,
		input_size._w
		);
}