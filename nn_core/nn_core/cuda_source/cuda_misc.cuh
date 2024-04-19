#ifndef _CUDA_MISC_CUH_
#define _CUDA_MISC_CUH_

#include "../cpp_source/cuda_common.h"


void transpose(
	const nn_type* input, 
	nn_type* output,
	cuint n,
	cuint c,
	cuint h,
	cuint w
);

void padding_dilation(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	cuint c,
	cuint in_h,
	cuint in_w,
	cuint out_h,
	cuint out_w,
	cuint offset_x,
	cuint offset_y,
	cuint stride_x,
	cuint stride_y
);

void add_bias_1d(
	const nn_type* input,
	const nn_type* bias,
	nn_type* output,
	cuint n,
	cuint c
);

void add_bias_2d(
	NN_Stream& s,
	const nn_type* input,
	const nn_type* bias,
	nn_type* output,
	cuint n,
	cuint c,
	cuint h,
	cuint w
);

void sum_gradient_1d(
	const nn_type* input,
	nn_type* output,
	cuint n,
	cuint c
);

void sum_gradient_2d(
	const nn_type* input,
	nn_type* output,
	cuint n,
	cuint c,
	cuint h,
	cuint w
);

#endif // !_CUDA_MISC_CUH_
