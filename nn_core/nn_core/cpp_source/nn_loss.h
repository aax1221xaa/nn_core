#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"


class NN_Loss {
public:
	std::string name;

	NN_Tensor* output;
	NN_Tensor d_output;

	NN_Loss();
};