#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "../cpp_source/nn_tensor.h"


void sgd(
	cudaStream_t stream,
	const CudaTensor gradient,
	CudaTensor momentum,
	CudaTensor weight,
	float learn_rate,
	float momentum_rate
);

void rms_prop(
	cudaStream_t stream,
	const CudaTensor gradient,
	CudaTensor square_g,
	CudaTensor weight,
	float decay_rate,
	float learn_rate
);

void adam(
	cudaStream_t stream,
	const CudaTensor gradient,
	CudaTensor square_g,
	CudaTensor decay_g,
	CudaTensor weight,
	float learn_rate,
	float beta_1,
	float beta_2
);

#endif // !OPTIMIZER_CUH
