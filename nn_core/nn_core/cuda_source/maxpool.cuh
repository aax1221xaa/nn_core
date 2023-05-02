#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

#include "../cpp_source/nn_tensor.h"

void maxpool_2d(
	cudaStream_t* stream,
	CudaTensor input,
	CudaTensor output,
	int kernel_w,
	int kernel_h,
	int stride_w,
	int stride_h
);


#endif // !MAXPOOL_CUH
