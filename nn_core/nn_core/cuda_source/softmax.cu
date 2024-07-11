#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "softmax.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __softmax(
	const nn_type* input,
	nn_type* output,
	cint c
) {
	__shared__ nn_type smem[BLOCK_1024];

	smem[threadIdx.x] = 0.f;
	__syncthreads();

	for (int i = 0; i < c; i += BLOCK_1024) {
		cint cx = i + threadIdx.x;

		smem[threadIdx.x] += cx < c ? __expf(input[cx]) : 0.f;
		__syncthreads();
	}

#pragma unroll
	for (int i = (BLOCK_1024 / 2); i > 0; i /= 2) {
		if (threadIdx.x < i) {
			smem[threadIdx.x] += smem[threadIdx.x + i];
		}
		__syncthreads();
	}

	for (int i = 0; i < c; i += BLOCK_1024) {
		cint cx = i + threadIdx.x;

		if (cx < c) output[cx] = __expf(input[cx]) / smem[0];
	}
}


/**********************************************/
/*                                            */
/*                  NN_Softmax                */
/*                                            */
/**********************************************/

NN_Softmax::NN_Softmax(const char* name) :
	NN_Layer(name)
{
}

void NN_Softmax::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_Softmax::build(const NN_List<NN_Shape>& input_shape, std::vector<GpuTensor<nn_type>>& weights) {

}

void NN_Softmax::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const NC in = input[0].val().get_shape().get_nc();

	const nn_type* in_data = input[0].val().get_ptr();
	nn_type* out_data = output[0].val().get_ptr();


	dim3 threads(BLOCK_1024);
	dim3 blocks(1);

	for (int n = 0; n < in.n; ++n) {
		const nn_type* m_in_data = in_data + (n * in.c);
		nn_type* m_out_data = out_data + (n * in.c);

		__softmax<<<blocks, threads, 0, st[n % STREAMS]>>>(
			m_in_data,
			m_out_data,
			in.c
		);
	}
}

NN_Backward* NN_Softmax::create_backward(NN_Optimizer& optimizer, std::vector<bool>& mask) {
	return new NN_dSoftmax(*this, optimizer);
}


/**********************************************/
/*                                            */
/*                 NN_dSoftmax                */
/*                                            */
/**********************************************/

NN_dSoftmax::NN_dSoftmax(NN_Softmax& softmax, NN_Optimizer& optimizer) :
	NN_Backward(optimizer),
	_softmax(softmax)
{
}

void NN_dSoftmax::get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape) {

}

void NN_dSoftmax::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}