#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"
#include "nn_ptr.h"


class NN_Layer {
public:
	string name;

	NN_Layer();
	~NN_Layer();

	virtual void calculate_output_size(vector<Dim*>& input_shape, Dim& output_shape) = 0;
	virtual void build(vector<Dim*>& input_shape);
	virtual void run_forward(vector<NN_Ptr<NN_Tensor>>& input, NN_Ptr<NN_Tensor>& output) = 0;
	virtual void run_backward(vector<NN_Ptr<NN_Tensor>>& d_output, NN_Ptr<NN_Tensor>& d_input) = 0;
};