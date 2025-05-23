#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "sigmoid.cuh"
/*
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>
*/
#include <cuda_runtime_api.h>
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


NN_Sigmoid::NN_Sigmoid(const std::string& name) :
	NN_Layer(name, "sigmoid")
{
}

void NN_Sigmoid::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_Sigmoid::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}

void NN_Sigmoid::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	cuint size = (cuint)input[0].val().get_shape().total_size();

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, size);

	__sigmoid<<<blocks, threads>>>(
		input[0].val().get_ptr(),
		output[0].val().get_ptr(),
		size
		);
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

NN_Backward* NN_Sigmoid::create_backward(std::vector<bool>& mask) {
	return new NN_dSigmoid(*this);
}


/**********************************************/
/*                                            */
/*                  NN_dSigmoid               */
/*                                            */
/**********************************************/

NN_dSigmoid::NN_dSigmoid(NN_Sigmoid& layer) :
	NN_Backward_t(layer)
{
}

void NN_dSigmoid::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}