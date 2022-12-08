#pragma once
#include "nn_manager.h"


class _Input : public NN_Layer {
public:
	_Input(Dim& input_size, int batch, const string layer_name);

	const Dim calculate_output_size(const vector<Dim>& input_size);
	void build(const vector<Dim>& input_size);
	void run_forward(const vector<NN_Tensor>& inputs, NN_Tensor& output);
	void run_backward(const vector<NN_Tensor>& d_outputs, NN_Tensor& d_input);
};


NN_Link* Input(Dim& input_size, int batch, const string layer_name = "");