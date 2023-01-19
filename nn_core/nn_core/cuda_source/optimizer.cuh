#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "../cpp_source/nn_tensor.h"


void sgd(
	cudaStream_t stream,
	const NN_Tensor4D gradient,
	NN_Tensor4D momentum,
	NN_Tensor4D weight,
	float learn_rate,
	float momentum_rate
);

void rms_prop(
	cudaStream_t stream,
	const NN_Tensor4D gradient,
	NN_Tensor4D square_g,
	NN_Tensor4D weight,
	float decay_rate,
	float learn_rate
);

void adam(
	cudaStream_t stream,
	const NN_Tensor4D gradient,
	NN_Tensor4D square_g,
	NN_Tensor4D decay_g,
	NN_Tensor4D weight,
	float learn_rate,
	float beta_1,
	float beta_2
);

#endif // !OPTIMIZER_CUH
