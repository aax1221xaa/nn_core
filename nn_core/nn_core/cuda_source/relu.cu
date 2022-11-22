#include "relu.cuh"

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

__global__ void __relu(
	float* a,
	float* b,
	const uint length
) {
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (cx < length) {
		b[cx] = __max(0.f, a[cx]);
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void relu(
	const Stream* stream,
	const Tensor* input,
	Tensor* output
) {
	size_t input_size = GetTotalSize(input);
	size_t output_size = GetTotalSize(output);

	if (input_size != output_size) {
		ErrorExcept("[relu] input과 output 사이즈가 안맞습니다. %d != %d", input_size, output_size);
	}

	int length = input->h * input->w * input->c;
	dim3 threads(BLOCK_SIZE);
	dim3 blocks(GetBlockSize(length));

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = input->data + (i * length);
		float* d_out = output->data + (i * length);

		__relu<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			d_out,
			length
		);
	}

	SyncStreams(stream);
}