#include "dens.cuh"

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
	float* a,
	float* b,
	float* c,
	const uint m,
	const uint n,
	const uint k
) {
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;
	uint cy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float sm_a[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sm_b[BLOCK_SIZE * BLOCK_SIZE];

	uint tidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;
	float val = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE) {
		__syncthreads();

		sm_a[tidx] = (threadIdx.x + i) < n && cy < m ? a[cy * n + (threadIdx.x + i)] : 0.f;
		sm_b[tidx] = cx < k && (threadIdx.y + i) < n ? b[(threadIdx.y + i) * k + cx] : 0.f;

		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; ++e) {
			val += sm_a[threadIdx.y * BLOCK_SIZE + e] * sm_b[e * BLOCK_SIZE + threadIdx.x];
		}
	}

	if (cx < k && cy < m) {
		c[cy * k + cx] = val;
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

//void check_dens(
//	const NN_Tensor4D input,
//	const NN_Tensor4D weight,
//	const NN_Tensor4D output
//) {
//	if (input.n != output.n || input.c != weight.n || output.c != weight.c) {
//		ErrorExcept(
//			"[matmul_check] invalid matrix size input: %s, weight: %s, output: %s",
//			dim_to_str(input),
//			dim_to_str(weight),
//			dim_to_str(output)
//		);
//	}
//}

void dens(
	const cudaStream_t st,
	const NN_Tensor4D input,
	const NN_Tensor4D weight,
	NN_Tensor4D output
) {
	//check_dens(input, weight, output);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, output.c, output.n);

	__matmul << <blocks, threads, 0, st >> > (
		input.data,
		weight.data,
		output.data,
		input.n,
		input.c,
		output.c
	);

	check_cuda(cudaStreamSynchronize(st));
}