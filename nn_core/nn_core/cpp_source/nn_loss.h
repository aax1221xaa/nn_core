#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"


class NN_Loss {
public:
	const char* name;

	GpuTensor<nn_type>* output;
	GpuTensor<nn_type> d_output;

	NN_Loss();
};