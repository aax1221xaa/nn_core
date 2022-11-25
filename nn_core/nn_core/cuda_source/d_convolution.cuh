#ifndef D_CONVOLUTION_CUH
#define D_CONVOLUTION_CUH

#include "../cpp_source/cuda_common.h"



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

#if false

void correl_2d(
	const Stream& stream,
	const Tensor& d_doutput,
	const Tensor& d_kernel,
	Tensor& d_dinput
);

void dilation_2d(
	const Stream& stream,
	const Tensor& input,
	Tensor& output,
	uint scale,
	int offset_x,
	int offset_y
);

#endif

#endif // !D_CONVOLUTION_CUH
