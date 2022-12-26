#include "nn_layer.h"



NN_Input::NN_Input(const Dim& input_size, int batch, const string layer_name) {
	name = layer_name;
	m_shape = input_size;
	m_shape[0] = batch;
}

NN_Input::~NN_Input() {

}

void NN_Input::calculate_output_size(NN_Vec<Dim*> input_shape, Dim& output_shape) {
	output_shape = m_shape;
}

void NN_Input::build(Dim& input_shape) {

}

void NN_Input::run_forward(NN_Vec<NN_Tensor*> input, NN_Tensor& output) {

}

void NN_Input::run_backward(NN_Vec<NN_Tensor*> d_output, NN_Tensor& d_input) {

}


NN_Link& Input(const Dim& input_size, int batch, const string layer_name) {
	NN_Layer* layer = new NN_Input(input_size, batch, layer_name);
	NN_Link* link = new NN_Link(layer);

	NN_Manager::add_link(link);

	return *link;
}