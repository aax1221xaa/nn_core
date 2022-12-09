#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"


class NN_Layer {
public:
	string name;
	bool trainable;

	NN_Layer();
	virtual ~NN_Layer() = 0;

	virtual const Dim calculate_output_size(const vector<Dim>& input_size) = 0;
	virtual void build(const vector<Dim>& input_size) = 0;
	virtual void run_forward(const vector<NN_Ptr<NN_Tensor>>& inputs, NN_Ptr<NN_Tensor>& output) = 0;
	virtual void run_backward(const vector<NN_Ptr<NN_Tensor>>& d_outputs, NN_Ptr<NN_Tensor>& d_input) = 0;
};