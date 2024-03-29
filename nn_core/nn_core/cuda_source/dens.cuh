#ifndef DENSE_CUH
#define DENSE_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************

				  Mat multiple

**********************************************/

void matmul(
	const nn_type* input,
	const nn_type* weight,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& out_shape
);

#endif