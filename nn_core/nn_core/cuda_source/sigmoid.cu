#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "sigmoid.cuh"

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

__global__ void __sigmoid(
	const nn_type* a,
	nn_type* b,
	cuint length
) {
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;

	if (cx < length) {
		b[cx] = 1.f / (1.f + __expf(-a[cx]));
	}
}


NN_Sigmoid::NN_Sigmoid(const char* name) :
	NN_Layer(name)
{
}

void NN_Sigmoid::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	output_shape.push_back(input_shape[0]);
}

void NN_Sigmoid::build(const std::vector<NN_Shape>& input_shape) {

}

void NN_Sigmoid::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	cuint size = (cuint)input[0].get_shape().total_size();

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, size);

	__sigmoid<<<blocks, threads>>>(
		input[0].get_ptr(),
		output[0].get_ptr(),
		size
		);
}