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

void NN_Input::calculate_output_size(vector<NN_Shape*>& input_shape, NN_Shape& output_shape) {
	output_shape = m_shape;
}

void NN_Input::build(vector<NN_Shape*>& input_shape) {

}

void NN_Input::run_forward(vector<NN_Tensor*>& input, NN_Tensor& output) {

}

void NN_Input::run_backward(
	vector<NN_Tensor*>& input,
	NN_Tensor& d_output,
	NN_Tensor& d_input,
	NN_Tensor& output)
{

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

void NN_Test::calculate_output_size(vector<NN_Shape*>& input_shape, NN_Shape& output_shape) {

}

void NN_Test::build(vector<NN_Shape*>& input_shape) {

}

void NN_Test::run_forward(vector<NN_Tensor*>& input, NN_Tensor& output) {

}

void NN_Test::run_backward(
	vector<NN_Tensor*>& input,
	NN_Tensor& d_output,
	NN_Tensor& d_input,
	NN_Tensor& output) 
{

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
	NN_Layer(_layer_name), weight(GPU), bias(GPU)
{
	amounts = _amounts;
}

void NN_Dense::calculate_output_size(vector<NN_Shape*>& input_shape, NN_Shape& output_shape) {
	NN_Shape& in_shape = *(input_shape[0]);

	output_shape.set({ in_shape[0], amounts });
}

void NN_Dense::build(vector<NN_Shape*>& input_shape) {
	const NN_Shape& shape = *(input_shape[0]);
	// [batch, 784]
	// [784, amounts] + [amounts]
	// [batch, amounts]

	weight.create({ shape[1], amounts });
	bias.create({ amounts });

	set_uniform(weight);
	check_cuda(cudaMemset(bias.data, 0, sizeof(float) * bias.get_elem_size()));
}

void NN_Dense::run_forward(vector<NN_Tensor*>& input, NN_Tensor& output) {
	dens(stream, input[0]->get_4dtensor(), weight.get_4dtensor(), output.get_4dtensor());
	add_bias(stream, output.get_4dtensor(), bias.get_4dtensor(), output.get_4dtensor());
}

void NN_Dense::run_backward(
	vector<NN_Tensor*>& input,
	NN_Tensor& d_output,
	NN_Tensor& d_input,
	NN_Tensor& output)
{
	NN_Shape& w_shape = weight.shape;
	NN_Shape& in_shape = d_input.shape;

	NN_Tensor t_weight({ w_shape[1], w_shape[0] }, GPU);
	NN_Tensor t_input({ in_shape[1], in_shape[0] }, GPU);

	transpose(stream, weight.get_4dtensor(), t_weight.get_4dtensor());
	dens(stream, d_output.get_4dtensor(), t_weight.get_4dtensor(), d_input.get_4dtensor());

	
}

NN_Link& Dense(int amounts, const string& _layer_name) {

}