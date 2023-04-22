#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/nn_tensor.h"



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

//int get_output_size(
//	int input_size,
//	int kernel_size,
//	int pad_size,
//	int stride
//);

void conv_2d(
	cudaStream_t stream,
	const CudaTensor d_input,
	const CudaTensor d_kernel,
	CudaTensor d_output,
	int st_w,
	int st_h
);

void correl_2d(
	cudaStream_t stream,
	const CudaTensor d_doutput,
	const CudaTensor d_kernel,
	CudaTensor d_dinput
);

void transpose(
	cudaStream_t stream,
	const CudaTensor d_input,
	CudaTensor d_output
);

void dilation_2d(
	cudaStream_t stream,
	const CudaTensor d_input,
	CudaTensor d_output,
	uint scale,
	int offset_x,
	int offset_y
);

void kernel_conv_2d(
	cudaStream_t stream,
	const CudaTensor d_doutput,
	const CudaTensor d_input,
	CudaTensor d_gradient
);

#endif // !_CONVOLUTION_CUH_
