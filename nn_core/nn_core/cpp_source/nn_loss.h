#pragma once
#include "cuda_common.h"


class NN_Loss {
public:
	string name;

	NN_Tensor* output;
	NN_Tensor d_output;


};