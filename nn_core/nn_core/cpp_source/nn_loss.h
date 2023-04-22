#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"


class NN_Loss {
public:
	std::string name;

	NN_Tensor<nn_type>* output;
	NN_Tensor<nn_type> d_output;

	NN_Loss();
};