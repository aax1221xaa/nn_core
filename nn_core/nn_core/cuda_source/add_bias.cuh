#ifndef ADD_BIAS_CUH
#define ADD_BIAS_CUH

#include "../cpp_source/nn_tensor.h"


void add_bias(
	cudaStream_t stream,
	const CudaTensor input,
	const CudaTensor bias,
	CudaTensor output
);

#endif // !ADD_BIAS_CUH
