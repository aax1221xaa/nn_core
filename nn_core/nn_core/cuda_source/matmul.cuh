#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************

					matmul

**********************************************/

void matmul(
	cuint m,
	cuint k,
	cuint n,
	const nn_type* input,
	const nn_type* weight,
	nn_type* output
);

#endif