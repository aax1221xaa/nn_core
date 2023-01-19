#ifndef ADD_BIAS_CUH
#define ADD_BIAS_CUH

#include "../cpp_source/nn_tensor.h"


void add_bias(
	cudaStream_t stream,
	const NN_Tensor4D input,
	const NN_Tensor4D bias,
	NN_Tensor4D output
);

#endif // !ADD_BIAS_CUH
