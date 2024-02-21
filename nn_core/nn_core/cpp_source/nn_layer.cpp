#include "nn_layer.h"


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

NN_Input::NN_Input(const nn_shape& input_size, int batch, const char* _layer_name) :
	NN_Layer(_layer_name),
	_shape(input_size)
{
	try {
		_shape.push_back(batch);

		for (const List<int>& n : _shape) {
			if (n._val < -1 || n._val == 0) {
				ErrorExcept(
					"[NN_Input::NN_Input] can't create input layer by dimension(%s).",
					put_shape(input_size)
				);
			}
		}
		
	}
	catch (const Exception& e) {
		NN_Manager::condition = false;
		e.Put();
	}
}

NN_Input::~NN_Input() {

}

nn_shape NN_Input::calculate_output_size(nn_shape& input_shape) {
	return { _shape };
}

void NN_Input::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {

}

void NN_Input::run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

NN_BackPropLayer* NN_Input::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}


Layer_t Input(const nn_shape& input_size, int batch, const char* layer_name) {
	NN_Input* layer = NULL;
	NN_Link* node = NULL;

	try {
		layer = new NN_Input(input_size, batch, layer_name);
		node = new NN_Link;

		if (!NN_Manager::condition) {
			ErrorExcept(
				"[Input()] can't create %s layer.",
				layer->_layer_name
			);
		}

		node->_forward = layer;

		NN_Manager::add_node(node);
		NN_Manager::add_layer(layer);
	}
	catch (const Exception& e) {
		delete layer;
		delete node;

		throw e;
	}

	return NN_Link::NN_LinkPtr({ 0, node });
}

/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

NN_Dense::NN_Dense(const int amounts, const char* name) :
	NN_Layer(name),
	_amounts(amounts)
{
}

nn_shape NN_Dense::calculate_output_size(nn_shape& input_shape) {
	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/
	return { {input_shape[0][0], _amounts} };
}

void NN_Dense::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {

}

void NN_Dense::run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) {

}

NN_BackPropLayer* NN_Dense::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}