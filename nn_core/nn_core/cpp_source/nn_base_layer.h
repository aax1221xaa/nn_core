#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"


class NN_Layer {
public:
	const string layer_name;

	NN_Layer(const string& _layer_name);
	~NN_Layer();

	virtual void calculate_output_size(vector<NN_Shape*>& input_shape, NN_Shape& output_shape) = 0;
	virtual void build(vector<NN_Shape*>& input_shape);
	virtual void run_forward(vector<NN_Tensor*>& input, NN_Tensor& output) = 0;
	virtual void run_backward(vector<NN_Tensor*>& d_output, NN_Tensor& d_input) = 0;
};

