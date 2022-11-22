#ifndef K_CONVOLUTION_CUH
#define K_CONVOLUTION_CUH

#include "../cpp_source/cuda_common.h"


#if _DEBUG

void kernel_conv_2d(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
);

__declspec(deprecated) void kernel_conv_2d_1x1024_g_ind(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
);

void kernel_conv_2d_32x32_c_ind(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
);

void kernel_conv_2d_32x32_g_ind(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
);

#else

void kernel_conv_2d(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
);

#endif

#endif // !K_CONVOLUTION_CUH
