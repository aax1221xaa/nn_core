#include "dens.cuh"
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
	const float* a,
	const float* b,
	float* c,
	const uint m,
	const uint n,
	const uint k
) {
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;
	uint cy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float sm_a[BLOCK_32 * BLOCK_32];
	__shared__ float sm_b[BLOCK_32 * BLOCK_32];

	uint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;
	float val = 0.f;

	for (int i = 0; i < n; i += BLOCK_32) {
		__syncthreads();

		sm_a[tidx] = (threadIdx.x + i) < n && cy < m ? a[cy * n + (threadIdx.x + i)] : 0.f;
		sm_b[tidx] = cx < k && (threadIdx.y + i) < n ? b[(threadIdx.y + i) * k + cx] : 0.f;

		__syncthreads();

#pragma unroll
		for (int e = 0; e < BLOCK_32; ++e) {
			val += sm_a[threadIdx.y * BLOCK_32 + e] * sm_b[e * BLOCK_32 + threadIdx.x];
		}
	}

	if (cx < k && cy < m) {
		c[cy * k + cx] = val;
	}
}

/**********************************************

				  FullyConnect

**********************************************/

void dense(
	const nn_type* input,
	const nn_type* weight,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& out_shape
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out_shape[1], out_shape[0]);

	__matmul<<<blocks, threads>>>(
		input,
		weight,
		output,
		in_shape[0],
		in_shape[1],
		out_shape[1]
	);
}