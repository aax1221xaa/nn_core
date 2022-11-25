#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/cuda_common.h"


void relu(
	const Stream& stream,
	const Tensor& input,
	Tensor& output
);


#endif // !RELU_CUH
