#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"


class NN_Layer {
public:
	const string layer_name;
	static cudaStream_t stream;
	static NN_Optimizer* optimizer;

	NN_Layer(const string& _layer_name);
	virtual ~NN_Layer();

	virtual void calculate_output_size(vector<NN_Shape_t>& input_shape, NN_Shape& output_shape) = 0;
	virtual void build(vector<NN_Shape_t>& input_shape);
	virtual void run_forward(vector<NN_Tensor_t>& input, NN_Tensor& output) = 0;
	virtual void run_backward(vector<NN_Tensor_t>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor_t>& d_input) = 0;
};

typedef NN_Layer* NN_Layer_t;


struct NN_Container {
	vector<NN_Tensor_t> input;
	NN_Tensor output;

	vector<NN_Tensor_t> d_input;
	vector<NN_Tensor_t> d_output;

	vector<NN_Shape_t> in_shape;
	NN_Shape out_shape;

	NN_Layer_t op_layer;
};

typedef NN_Container* NN_Container_t;