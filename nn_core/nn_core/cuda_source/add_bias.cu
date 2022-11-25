#include "add_bias.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __add_bias(
	float* a,
	float* b,
	float* c,
	uint len,
	uint ch
) {
	extern __shared__ float sm_b[];
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();

	for (uint i = 0; i < ch; i += BLOCK_SIZE_32) {
		int tidx = threadIdx.x + i;

		if (tidx < ch) {
			sm_b[tidx] = b[tidx];
		}
	}

	__syncthreads();

	for (uint i = 0; i < ch; ++i) {
		float* pa = a + (len * i);
		float* pc = c + (len * i);

		if (cx < len) {
			pc[cx] = pa[cx] + sm_b[i];
		}
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void check_add_bias(
	const Tensor& input,
	const Tensor& bias,
	const Tensor& output
) {
	int input_len = input.h * input.w;
	int output_len = output.h * output.w;

	if (input_len != output_len || input.c != output.c || bias.w != output.c) {
		ErrorExcept(
			"[check_add_bias] invalid size. input: [%d, %d, %d], bias: [%d], output: [%d, %d, %d]",
			input.c, input.h, input.w, bias.w, output.c, output.h, output.w
		);
	}
}

void add_bias(
	const Stream& stream,
	const Tensor& input,
	const Tensor& bias,
	Tensor& output
) {
	check_add_bias(input, bias, output);

	uint length = output.h * output.w;
	size_t share_size = sizeof(float) * output.c;

	dim3 threads(BLOCK_SIZE_1024);
	dim3 blocks = get_grid_size(threads, length);

	for (int i = 0; i < stream.str_size; ++i) {
		float* p_input = input.data + (i * length * input.c);
		float* p_output = output.data + (i * length * output.c);

		__add_bias << <blocks, threads, share_size, stream.str[i] >> > (
			p_input,
			bias.data,
			p_output,
			length,
			output.c
		);
	}

	sync_streams(stream);
}