#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "../cpp_source/cuda_common.h"


void sgd(
	Stream& stream,
	const Tensor& gradient,
	Tensor& momentum,
	Tensor& weight,
	float learn_rate,
	float momentum_rate
);

void rms_prop(
	Stream& stream,
	const Tensor& gradient,
	Tensor& square_g,
	Tensor& weight,
	float decay_rate,
	float learn_rate
);

void adam(
	Stream& stream,
	const Tensor& gradient,
	Tensor& square_g,
	Tensor& decay_g,
	Tensor& weight,
	float learn_rate,
	float beta_1,
	float beta_2
);

#endif // !OPTIMIZER_CUH
