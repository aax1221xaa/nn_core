#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
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

NN_ReLU::NN_ReLU(const std::string& name) :
	NN_Layer(name)
{
}

void NN_ReLU::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_ReLU::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}

void NN_ReLU::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	cuint size = (cuint)input[0].val().get_shape().total_size();

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, size);

	__relu<<<blocks, threads>>>(
		input[0].val().get_ptr(),
		output[0].val().get_ptr(),
		size
	);

	//check_cuda(cudaDeviceSynchronize());
	//check_cuda(cudaGetLastError());
}

NN_Backward* NN_ReLU::create_backward(std::vector<bool>& mask) {
	return new NN_dReLU(*this);
}


/**********************************************/
/*                                            */
/*                   NN_dReLU                 */
/*                                            */
/**********************************************/

NN_dReLU::NN_dReLU(NN_ReLU& relu) :
	_relu(relu)
{

}

void NN_dReLU::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}