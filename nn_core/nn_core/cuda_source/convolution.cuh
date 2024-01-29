#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/cuda_common.h"


/**********************************************

				    Conv2d

**********************************************/

void conv2d(
	cudaStream_t s,
	cuint* indice,
	const nn_type* input,
	const nn_type* kernel,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& k_shape,
	const nn_shape& out_shape,
	cuint h_stride,
	cuint w_stride
);

/**********************************************

		         KernelConv2d

**********************************************/

void kernel_conv2d(
	const nn_type* d_output,
	const nn_type* input,
	nn_type* grad,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	const nn_shape& grad_shape
);

#endif // !_CONVOLUTION_CUH_