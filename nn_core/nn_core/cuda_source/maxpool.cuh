#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************

				    Maxpool2d

**********************************************/

void maxpool2d(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	uint* max_indice,
	cuint in_h,
	cuint in_w,
	cuint out_h,
	cuint out_w,
	cuint h_kernel,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride,
	cuint h_tile,
	cuint w_tile
);

/**********************************************

				  D_Maxpool2d

**********************************************/
/*
void d_maxpool2d(
	cudaStream_t* s,
	const nn_type* d_output,
	nn_type* d_input,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	cuint* max_indice,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride
);
*/

#endif // !MAXPOOL_CUH
