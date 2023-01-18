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

void check_dens(
	const NN_Tensor& input,
	const NN_Tensor& weight,
	const NN_Tensor& output
) {
	const NN_Shape& in_shape = input.shape;
	const NN_Shape& w_shape = weight.shape;
	const NN_Shape& out_shape = output.shape;

	if (input.device_type != GPU || weight.device_type != GPU || output.device_type != GPU) {
		ErrorExcept(
			"[check_dense] Tensor is not GPU Memory"
		);
	}
	if (in_shape.len != 2 || w_shape.len != 2 || out_shape.len != 2) {
		ErrorExcept(
			"[check_dense] invalid matrix shapes input: %s, weight: %s, output: %s",
			in_shape.get_str(),
			w_shape.get_str(),
			out_shape.get_str()
		);
	}
	if (in_shape[0] != out_shape[0] || in_shape[-1] != w_shape[0] || out_shape[-1] != w_shape[-1]) {
		ErrorExcept(
			"[check_dense] invalid matrix size input: %s, weight: %s, output: %s",
			in_shape.get_str(),
			w_shape.get_str(),
			out_shape.get_str()
		);
	}
}

void dens(
	const cudaStream_t st,
	const NN_Tensor& input,
	const NN_Tensor& weight,
	NN_Tensor& output
) {
	check_dens(input, weight, output);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, output.shape[-1], output.shape[0]);

	__matmul<<<blocks, threads, 0, st>>>(
		input.data,
		weight.data,
		output.data,
		input.shape[0],
		input.shape[-1],
		output.shape[-1]
	);

	check_cuda(cudaStreamSynchronize(st));
}