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

	for (uint i = 0; i < ch; i += BLOCK_SIZE) {
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
	const Tensor* input,
	const Tensor* bias,
	const Tensor* output
) {
	int input_len = input->h * input->w;
	int output_len = output->h * output->w;

	if (input_len != output_len || input->c != output->c || bias->w != output->c) {
		ErrorExcept(
			"[check_add_bias] invalid size. input: [%d, %d, %d], bias: [%d], output: [%d, %d, %d]",
			input->h, input->w, input->c, bias->w, output->h, output->w, output->c
		);
	}
}

void add_bias(
	const Stream* stream,
	const Tensor* input,
	const Tensor* bias,
	Tensor* output
) {
	check_add_bias(input, bias, output);

	uint length = output->h * output->w;
	size_t share_size = sizeof(float) * output->c;

	dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocks((length + threads.x - 1) / threads.x);

	for (int i = 0; i < stream->st_size; ++i) {
		float* p_input = input->data + (i * length * input->c);
		float* p_output = output->data + (i * length * output->c);

		__add_bias<<<blocks, threads, share_size, stream->st[i]>>>(
			p_input,
			bias->data,
			p_output,
			length,
			output->c
		);
	}

	SyncStreams(stream);
}