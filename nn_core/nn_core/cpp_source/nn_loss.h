#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"


class NN_Loss {
public:
	const char* name;

	DeviceTensor<nn_type>* output;
	DeviceTensor<nn_type> d_output;

	NN_Loss();
};