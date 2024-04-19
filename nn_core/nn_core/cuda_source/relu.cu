#include "relu.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

//#include <device_functions.h>
#include <device_launch_parameters.h>


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __relu(
	const nn_type* a,
	nn_type* b,
	cuint length
) {
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (cx < length) {
		b[cx] = __max(0.f, a[cx]);
	}
}

/*
__global__ void __d_relu(
	const float* a,
	const float* b,
	float* c,
	cuint len
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;

	if (cx < len && b[cx] > 0) c[cx] = a[cx];
}
*/

/**********************************************

					  ReLU

**********************************************/

void relu(
	const nn_type* input,
	nn_type* output,
	cuint len
) {
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__relu<<<blocks, threads>>>(
		input,
		output,
		len
	);
}