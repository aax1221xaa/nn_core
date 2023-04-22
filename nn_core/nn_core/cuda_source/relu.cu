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
	const float* a,
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
	const float* input,
	float* output,
	cuint len
) {
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks(get_grid_size(threads, len));

	__relu << <blocks, threads, 0, stream >> > (
		input,
		output,
		len
		);

	check_cuda(cudaStreamSynchronize(stream));
}