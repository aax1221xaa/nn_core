#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/nn_tensor.h"


void relu(
	cudaStream_t stream,
	const NN_Tensor4D input,
	NN_Tensor4D output
);


#endif // !RELU_CUH
