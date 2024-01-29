#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "../cpp_source/cuda_common.h"


void sgd(
	const nn_type* gradient,
	nn_type* momentum,
	nn_type* weight,
	const nn_shape& w_shape,
	float learn_rate,
	float momentum_rate
);

void rms_prop(
	const nn_type* gradient,
	nn_type* square_g,
	nn_type* weight,
	const nn_shape& w_shape,
	float decay_rate,
	float learn_rate
);

void adam(
	const nn_type* gradient,
	nn_type* square_g,
	nn_type* decay_g,
	nn_type* weight,
	const nn_shape& w_shape,
	float learn_rate,
	float beta_1,
	float beta_2
);

#endif // !OPTIMIZER_CUH
