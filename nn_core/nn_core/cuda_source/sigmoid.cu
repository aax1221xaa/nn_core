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

void NN_Sigmoid::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_Sigmoid::build(const NN_List<NN_Shape>& input_shape) {

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
}

NN_Backward* NN_Sigmoid::create_backward(NN_Optimizer* optimizer) {
	return new NN_dSigmoid(this, optimizer);
}


/**********************************************/
/*                                            */
/*                  NN_dSigmoid               */
/*                                            */
/**********************************************/

NN_dSigmoid::NN_dSigmoid(NN_Sigmoid* sigmoid, NN_Optimizer* optimizer) :
	NN_Backward(optimizer),
	_sigmoid(sigmoid)
{

}

void NN_dSigmoid::get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape) {

}

void NN_dSigmoid::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}