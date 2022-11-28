#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/cuda_common.h"



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

int get_output_size(
	int input_size,
	int kernel_size,
	int pad_size,
	int stride
);

void conv_2d(
	const Stream& stream,
	const Tensor& d_input,
	const Tensor& d_kernel,
	Tensor& d_output,
	int st_w,
	int st_h
);

void correl_2d(
	const Stream& stream,
	const Tensor& d_doutput,
	const Tensor& d_kernel,
	Tensor& d_dinput
);

void transpose(
	const Stream& stream,
	const Tensor& d_input,
	Tensor& d_output
);

void dilation_2d(
	const Stream& stream,
	const Tensor& d_input,
	Tensor& d_output,
	uint scale,
	int offset_x,
	int offset_y
);

void kernel_conv_2d(
	const Stream& stream,
	const Tensor& d_doutput,
	const Tensor& d_input,
	Tensor& d_gradient
);

#endif // !_CONVOLUTION_CUH_
