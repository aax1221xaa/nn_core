#include "matmul.cuh"
#include "cuda_misc.cuh"


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

__global__ void __matmul(
	const nn_type* a,
	const nn_type* b,
	nn_type* c,
	cuint m,
	cuint k,
	cuint n
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ nn_type sm_a[BLOCK_32 * BLOCK_32];
	__shared__ nn_type sm_b[BLOCK_32 * BLOCK_32];

	cuint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;
	nn_type val = 0.f;

	for (uint i = 0; i < k; i += BLOCK_32) {
		__syncthreads();

		sm_a[tidx] = (threadIdx.x + i) < k && cy < m ? a[cy * k + (threadIdx.x + i)] : 0.f;
		sm_b[tidx] = cx < n && (threadIdx.y + i) < k ? b[(threadIdx.y + i) * n + cx] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_32; ++e) {
			val += sm_a[threadIdx.y * BLOCK_32 + e] * sm_b[e * BLOCK_32 + threadIdx.x];
		}
	}

	if (cx < n && cy < m) {
		c[cy * n + cx] = val;
	}
}

/**********************************************

					  matmul

**********************************************/

void matmul(
	cuint m,
	cuint k,
	cuint n,
	const nn_type* input,
	const nn_type* weight,
	nn_type* output
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, n, m);

	__matmul<<<blocks, threads>>>(
		input,
		weight,
		output,
		m,
		k,
		n
	);
}