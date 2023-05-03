#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/nn_tensor.h"


/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void copy_to_indice(
	const uint* indice,
	const size_t size,
	const size_t offset
);

void conv_2d(
	cudaStream_t* streams,
	const CudaTensor d_input,
	const CudaTensor d_kernel,
	CudaTensor d_output,
	int st_w,
	int st_h,
	int indice_offset
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

void padding_conv_2d(
	cudaStream_t* s,
	const CudaTensor d_input,
	CudaTensor d_pad,
	const CudaTensor d_kernel,
	CudaTensor d_output,
	int st_w,
	int st_h,
	int indice_offset
);

void kernel_conv_2d(
	cudaStream_t stream,
	const CudaTensor d_doutput,
	const CudaTensor d_input,
	CudaTensor d_gradient
);


#endif // !_CONVOLUTION_CUH_