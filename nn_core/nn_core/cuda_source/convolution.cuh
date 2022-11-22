#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/cuda_common.h"


__constant__ uint __indices[CONST_SIZE];

/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __conv_2d(
	float* input,
	float* kernel,
	float* output,
	cuint in_w,
	cuint k_n,
	cuint k_w,
	cuint k_h,
	cuint k_c,
	cuint out_w,
	cuint out_h,
	cuint st_w,
	cuint st_h
);


/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

uint get_output_size(
	int input_size,
	int kernel_size,
	int pad_size,
	int stride
);

void check_conv_2d(
	const Tensor* d_input,
	const Tensor* d_kernel,
	const Tensor* d_output,
	int st_w,
	int st_h
);

void conv_2d(
	const Stream* stream,
	const Tensor* d_input,
	const Tensor* d_kernel,
	Tensor* d_output,
	int st_w,
	int st_h
);

#endif // !_CONVOLUTION_CUH_
