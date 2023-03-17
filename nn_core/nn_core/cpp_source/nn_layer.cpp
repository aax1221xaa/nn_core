#include "nn_layer.h"
#include "../cuda_source/dens.cuh"
#include "../cuda_source/add_bias.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/convolution.cuh"



NN_Input::NN_Input(const NN_Shape& input_size, int batch, const string _layer_name) :
	NN_Layer(_layer_name)
{
	m_shape = input_size;
	m_shape[0] = batch;
}

NN_Input::~NN_Input() {

}

void NN_Input::calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape) {

}

void NN_Input::build(vector<NN_Shape>& input_shape) {

}

void NN_Input::run_forward(vector<NN_Tensor>& input, NN_Tensor& output) {

}

void NN_Input::run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input) {

}


NN Input(const NN_Shape& input_size, int batch, const string layer_name) {
	NN_Layer* layer = new NN_Input(input_size, batch, layer_name);
	NN_Link* link = new NN_Link;

	link->op_layer = layer;

	NN_Manager::add_layer(layer);
	NN_Manager::add_link(link);

	return NN(link);
}


NN_Test::NN_Test(const string name) :
	NN_Layer(name)
{
}

void NN_Test::calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape) {

}

void NN_Test::build(vector<NN_Shape>& input_shape) {

}

void NN_Test::run_forward(vector<NN_Tensor>& input, NN_Tensor& output) {

}

void NN_Test::run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input) {

}

NN_Link& Test(const string name) {
	NN_Layer* layer = new NN_Test(name);
	NN_Link* link = new NN_Link;

	link->op_layer = layer;

	NN_Manager::add_layer(layer);
	NN_Manager::add_link(link);

	return *link;
}


NN_Dense::NN_Dense(int _amounts, const string& _layer_name) :
	NN_Layer(_layer_name)
{
	amounts = _amounts;
}

void NN_Dense::calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape) {
	NN_Shape& in_shape = input_shape[0];

	output_shape.set({ in_shape[0], amounts });
}

void NN_Dense::build(vector<NN_Shape>& input_shape) {
	const NN_Shape& shape = input_shape[0];
	// [batch, 784]
	// [784, amounts] + [amounts]
	// [batch, amounts]

	weight.set({ shape[1], amounts }, GPU);
	bias.set({ amounts }, GPU);

	set_uniform(weight);
	check_cuda(cudaMemset(bias.data, 0, bias.bytes));
}

void NN_Dense::run_forward(vector<NN_Tensor>& input, NN_Tensor& output) {

}

void NN_Dense::run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input) {

}

NN_Link& Dense(int amounts, const string& _layer_name) {
	NN_Layer* layer = new NN_Dense(amounts, _layer_name);
	NN_Link* link = new NN_Link;

	link->op_layer = layer;

	NN_Manager::add_layer(layer);
	NN_Manager::add_link(link);

	return *link;
}