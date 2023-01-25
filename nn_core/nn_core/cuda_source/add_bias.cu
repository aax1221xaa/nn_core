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
	cuint node_size,
	cuint channels
) {
	cuint index = blockIdx.x * blockDim.x + threadIdx.x;
	cuint n_channel = index / node_size;

	if (n_channel < channels) {
		c[index] = a[index] + b[n_channel];
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
	const NN_Tensor4D input,
	const NN_Tensor4D bias,
	NN_Tensor4D output
) {
	//check_add_bias(input, bias, output);

	uint node_size = output.h * output.w;
	uint channel_size = output.c;

	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, output.c * output.h * output.w);

	for (int i = 0; i < input.n; ++i) {
		float* p_input = input.data + (i * input.h * input.w * input.c);
		float* p_output = output.data + (i * output.h * output.w * output.c);

		__add_bias << <blocks, threads, 0, stream >> > (
			p_input,
			bias.data,
			p_output,
			node_size,
			channel_size
		);
	}

	check_cuda(cudaStreamSynchronize(stream));
}