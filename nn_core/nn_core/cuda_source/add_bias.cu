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

__global__ void __add_bias_32x32(
	const float* data_a,
	const float* data_b,
	float* data_c,
	cuint n,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;

	cuint addr = cy * c + cx;

	__shared__ float share_b[BLOCK_SIZE];

	
	if (threadIdx.y == 0) share_b[threadIdx.x] = cx < c ? data_b[cx] : 0.f;
	__syncthreads();

	if (cx < c && cy < n) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.x];
	}
	__syncthreads();
}

__global__ void __add_bias_16x16x4(
	const float* data_a,
	const float* data_b,
	float* data_c,
	cuint c,
	cuint h,
	cuint w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z * blockDim.z + threadIdx.z;

	cuint addr = cz * (h * w) + cy * w + cx;

	__shared__ float share_b[SMALL_Z];

	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0) share_b[threadIdx.z] = cz < c ? data_b[cz] : 0.f;
	__syncthreads();

	if (cx < w && cy < h && cz < c) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.z];
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

//void check_add_bias(
//	const NN_Tensor4D input,
//	const NN_Tensor4D bias,
//	const NN_Tensor4D output
//) {
//	int in_node_size = input.h * input.w;
//	int out_node_size = output.h * output.w;
//
//	if (in_node_size != out_node_size || 
//		input.c != output.c || 
//		bias.n != output.c ||
//		get_elem_size(input) != get_elem_size(output)
//		) {
//		ErrorExcept(
//			"[check_add_bias] invalid size. input: %s, bias: %s, output: %s",
//			dim_to_str(input),
//			dim_to_str(bias),
//			dim_to_str(output)
//		);
//	}
//}

void add_bias(
	cudaStream_t stream,
	const CudaTensor input,
	const CudaTensor bias,
	CudaTensor output
) {
	//check_add_bias(input, bias, output);
	check_cuda(cudaGetLastError());

	if (input.h == 1 && input.w == 1) {
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocks = get_grid_size(threads, input.c, input.n);

		__add_bias_32x32<<<blocks, threads, 0, stream>>>(
			input.data,
			bias.data,
			output.data,
			input.n,
			input.c
		);
	}
	else {
		dim3 threads(SMALL_XY, SMALL_XY, SMALL_Z);
		dim3 blocks = get_grid_size(threads, input.w, input.h, input.c);

		for (uint i = 0; i < input.n; ++i) {
			float* da = input.data + (i * (input.c * input.h * input.w));
			float* db = bias.data;
			float* dc = output.data + (i * (output.c * output.h * output.w));

			__add_bias_16x16x4<<<blocks, threads, 0, stream>>> (
				da,
				db,
				dc,
				input.c,
				input.h,
				input.w
			);
		}
	}

	//check_cuda(cudaStreamSynchronize(stream));
	//check_cuda(cudaGetLastError());
}