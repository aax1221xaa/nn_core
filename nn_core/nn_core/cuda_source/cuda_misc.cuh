#ifndef _CUDA_MISC_CUH_
#define _CUDA_MISC_CUH_

#include "../cpp_source/nn_tensor.h"


void transpose(
	const GpuTensor<nn_type>& input, 
	GpuTensor<nn_type>& output
);

void padding_dilation(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	const NCHW in,
	const NCHW out,
	cuint offset_x,
	cuint offset_y,
	cuint stride_x,
	cuint stride_y
);

void add_bias_1d(
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
);

void add_bias_2d(
	NN_Stream& s,
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
);

void sum_gradient_1d(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output
);

void sum_gradient_2d(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output
);

#endif // !_CUDA_MISC_CUH_
