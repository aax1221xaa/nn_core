#ifndef K_CONVOLUTION_CUH
#define K_CONVOLUTION_CUH

#include "../cpp_source/cuda_common.h"

#if false

void kernel_conv_2d(
	const Stream& stream,
	const Tensor& d_input,
	const Tensor& d_output,
	Tensor& gradient
);

#endif

#endif // !K_CONVOLUTION_CUH
