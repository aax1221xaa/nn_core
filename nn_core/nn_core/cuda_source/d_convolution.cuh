#ifndef D_CONVOLUTION_CUH
#define D_CONVOLUTION_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __transpose(
	float* input,
	float* output,
	cuint n,
	cuint h,
	cuint w,
	cuint c
);

__global__ void __dilation_2d(
	float* input,
	float* output,
	cuint iw,
	cuint ih,
	cuint ic,
	cuint ow,
	cuint oh,
	cint scale,
	cint offset_x,
	cint offset_y
);


/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void correl_2d(
	const Stream* stream,
	const Tensor* d_output,
	const Tensor* kernel,
	Tensor* d_input,
	Tensor* work_space
);

void dilation_2d(
	const Stream* stream,
	const Tensor* input,
	Tensor* output,
	int scale,
	int offset_x,
	int offset_y
);

#endif // !D_CONVOLUTION_CUH
