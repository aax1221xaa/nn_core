#ifndef _CUDA_MISC_CUH_
#define _CUDA_MISC_CUH_

#include "../cpp_source/nn_tensor.h"
#include "../cpp_source/gpu_tensor.h"


void set_const_mem(cuint* h_mem, size_t len, size_t offset);
cuint* get_const_mem(size_t len, size_t offset);

void transpose(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output,
	const std::vector<uint>& ranks
);

void padding_dilation(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	const NN_Tensor4dShape& in,
	const NN_Tensor4dShape& out,
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
