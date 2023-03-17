#pragma once
#include "nn_manager.h"


class NN_Input : public NN_Layer {
public:
	NN_Shape m_shape;

	NN_Input(const NN_Shape& input_size, int batch, const string _layer_name);
	virtual ~NN_Input();

	void calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape);
	void build(vector<NN_Shape>& input_shape);
	void run_forward(vector<NN_Tensor>& input, NN_Tensor& output);
	void run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input);
};


NN Input(const NN_Shape& input_size, int batch, const string layer_name = "");


class NN_Test : public NN_Layer {
public:
	NN_Test(const string name);
	
	void calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape);
	void build(vector<NN_Shape>& input_shape);
	void run_forward(vector<NN_Tensor>& input, NN_Tensor& output);
	void run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input);
};

NN_Link& Test(const string name);


class NN_Dense : public NN_Layer {
public:
	NN_Tensor weight;
	NN_Tensor bias;
	int amounts;

	NN_Dense(int _amounts, const string& _layer_name);

	void calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape);
	void build(vector<NN_Shape>& input_shape);
	void run_forward(vector<NN_Tensor>& input, NN_Tensor& output);
	void run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input);
};

NN_Link& Dense(int amounts, const string& _layer_name);