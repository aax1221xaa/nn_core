#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************

					  ReLU

**********************************************/

void relu(
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape
);

/**********************************************

					 D_ReLU

**********************************************/

void d_relu(
	const nn_type* d_output,
	const nn_type* input,
	nn_type* d_input,
	const nn_shape& in_shape
);


#endif // !RELU_CUH
