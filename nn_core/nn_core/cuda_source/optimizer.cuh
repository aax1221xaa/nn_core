#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "../cpp_source/cuda_common.h"


void sgd(
	nn_type* gradient,
	nn_type* momentum,
	nn_type* weight,
	const uint len,
	float learn_rate,
	float momentum_rate
);

void rms_prop(
	nn_type* gradient,
	nn_type* square_g,
	nn_type* weight,
	const uint len,
	float decay_rate,
	float learn_rate
);

void adam(
	nn_type* gradient,
	nn_type* square_g,
	nn_type* decay_g,
	nn_type* weight,
	const uint len,
	float learn_rate,
	float beta_1,
	float beta_2
);

#endif // !OPTIMIZER_CUH
