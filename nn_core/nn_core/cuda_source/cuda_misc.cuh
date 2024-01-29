#ifndef _CUDA_MISC_CUH_
#define _CUDA_MISC_CUH_

#include "../cpp_source/cuda_common.h"


void transpose(
	const nn_type* input, 
	nn_type* output,
	const nn_shape& in_shape
);

void padding_dilation(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& out_shape,
	int offset_x,
	int offset_y,
	int stride_x,
	int stride_y
);

void add_bias_1d(
	const nn_type* input,
	const nn_type* bias,
	nn_type* output,
	const nn_shape& in_shape
);

void add_bias_2d(
	cudaStream_t* s,
	const nn_type* input,
	const nn_type* bias,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& b_shape,
	const nn_shape& out_shape
);

void sum_gradient_1d(
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape
);

void sum_gradient_2d(
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape
);

#endif // !_CUDA_MISC_CUH_
