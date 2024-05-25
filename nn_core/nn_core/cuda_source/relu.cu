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

/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

NN_ReLU::NN_ReLU(const char* name) :
	NN_Layer(name)
{
}

void NN_ReLU::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	output_shape.push_back(input_shape[0]);
}

void NN_ReLU::build(const std::vector<NN_Shape>& input_shape) {

}

void NN_ReLU::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	cuint size = (cuint)input[0].get_shape().total_size();

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, size);

	__relu<<<blocks, threads>>>(
		input[0].get_ptr(),
		output[0].get_ptr(),
		size
	);

	//check_cuda(cudaDeviceSynchronize());
	//check_cuda(cudaGetLastError());
}