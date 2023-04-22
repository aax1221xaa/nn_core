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
	cudaStream_t stream,
	const CudaTensor input,
	CudaTensor output
) {
	uint input_size = get_elem_size(input);
	uint output_size = get_elem_size(output);

	if (input_size != output_size) {
		ErrorExcept("[relu] invalid input and output size. %d != %d", input_size, output_size);
	}

	uint length = input.h * input.w * input.c;
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks(get_grid_size(threads, length));

	for (int i = 0; i < input.n; ++i) {
		float* d_in = input.data + (i * length);
		float* d_out = output.data + (i * length);

		__relu<<<blocks, threads, 0, stream>>>(
			d_in,
			d_out,
			length
		);
	}
	check_cuda(cudaStreamSynchronize(stream));
}