#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

#include "../cpp_source/cuda_common.h"

int calc_output_size(
	int input_size,
	int k_size,
	int strides
);

void maxpool_2d(
	Stream& stream,
	Tensor& input,
	Tensor& output,
	int kernel_w,
	int kernel_h,
	int stride_w,
	int stride_h
);


#endif // !MAXPOOL_CUH
