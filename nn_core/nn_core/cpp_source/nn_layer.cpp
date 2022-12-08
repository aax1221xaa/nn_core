#include "nn_layer.h"



_Input::_Input(Dim& input_size, int batch, const string layer_name) {
	
}

const Dim _Input::calculate_output_size(const vector<Dim>& input_size) {

}

void _Input::build(const vector<Dim>& input_size) {

}

void _Input::run_forward(const vector<NN_Tensor>& inputs, NN_Tensor& output) {

}

void _Input::run_backward(const vector<NN_Tensor>& d_outputs, NN_Tensor& d_input) {

}


NN_Link* Input(Dim& input_size, int batch, const string layer_name = "") {
	NN_Layer* layer = new _Input(input_size, batch, layer_name);
	NN_Link* link = new NN_Link(layer);

	NN_Manager::add_layer(layer);
	NN_Manager::add_link(link);

	return link;
}