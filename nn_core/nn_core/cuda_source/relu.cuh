#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************

					  ReLU

**********************************************/

void relu(
	const nn_type* input,
	nn_type* output,
	cuint len
);



#endif // !RELU_CUH
