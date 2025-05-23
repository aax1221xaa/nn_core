#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "softmax.cuh"
/*
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>
*/

#include <cuda_runtime_api.h>
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

NN_Softmax::NN_Softmax(const std::string& name) :
	NN_Layer(name, "softmax")
{
}

void NN_Softmax::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_Softmax::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}

void NN_Softmax::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const NN_Shape in = input[0].val().get_shape();

	const nn_type* in_data = input[0].val().get_ptr();
	nn_type* out_data = output[0].val().get_ptr();


	dim3 threads(BLOCK_1024);
	dim3 blocks(1);

	for (int n = 0; n < in[0]; ++n) {
		const nn_type* m_in_data = in_data + (n * in[1]);
		nn_type* m_out_data = out_data + (n * in[1]);

		__softmax<<<blocks, threads, 0, st[n % STREAMS]>>>(
			m_in_data,
			m_out_data,
			in[1]
		);
	}
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

NN_Backward* NN_Softmax::create_backward(std::vector<bool>& mask) {
	return new NN_dSoftmax(*this);
}


/**********************************************/
/*                                            */
/*                 NN_dSoftmax                */
/*                                            */
/**********************************************/

NN_dSoftmax::NN_dSoftmax(NN_Softmax& layer) :
	NN_Backward_t(layer)
{
}

void NN_dSoftmax::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}