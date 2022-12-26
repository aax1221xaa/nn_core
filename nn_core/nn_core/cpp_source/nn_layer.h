#pragma once
#include "nn_manager.h"


class NN_Input : public NN_Layer {
public:
	Dim m_shape;

	NN_Input(const Dim& input_size, int batch, const string layer_name);
	~NN_Input();

	void calculate_output_size(NN_Vec<Dim*> input_shape, Dim& output_shape);
	void build(Dim& input_shape);
	void run_forward(NN_Vec<NN_Tensor*> input, NN_Tensor& output);
	void run_backward(NN_Vec<NN_Tensor*> d_output, NN_Tensor& d_input);
};


NN_Link& Input(const Dim& input_size, int batch, const string layer_name = "");