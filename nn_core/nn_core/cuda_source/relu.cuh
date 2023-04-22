#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/cuda_common.h"


void relu(
	cudaStream_t stream,
	const float* input,
	float* output,
	cuint len
);


#endif // !RELU_CUH
