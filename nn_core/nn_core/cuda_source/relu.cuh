#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/cuda_common.h"


void relu(
	cudaStream_t stream,
	const CudaTensor input,
	CudaTensor output
);


#endif // !RELU_CUH
