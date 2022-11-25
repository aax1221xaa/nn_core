#ifndef ADD_BIAS_CUH
#define ADD_BIAS_CUH

#include "../cpp_source/cuda_common.h"


void add_bias(
	const Stream& stream,
	const Tensor& input,
	const Tensor& bias,
	Tensor& output
);

#endif // !ADD_BIAS_CUH
