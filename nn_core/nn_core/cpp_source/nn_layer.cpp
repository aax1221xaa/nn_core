#include "nn_layer.h"
#include "../cuda_source/dens.cuh"
#include "../cuda_source/add_bias.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/convolution.cuh"


#ifdef FIX_MODE

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
		_shape.insert(_shape.begin(), batch);

		for (const int& n : _shape) {
			if (n < -1 || n == 0) {
				ErrorExcept(
					"[NN_Input::NN_Input] can't create input layer by dimension(%s).",
					dimension_to_str(input_size)
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

nn_shape NN_Input::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	if (input_shape.size() == 0) return _shape;
	else if (input_shape.size() > 1) {
		ErrorExcept(
			"[NN_Input::calculate_output_size()] input layer can't receive %d layers.",
			input_shape.size()
		);
	}

	return *input_shape[0];
}

void NN_Input::build(std::vector<nn_shape*>& input_shape) {

}

NN_Tensor<nn_type> NN_Input::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input) {
	return *input[0];
}

NN_Tensor<nn_type> NN_Input::run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output) {
	return NN_Tensor<nn_type>();
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

	return { Layer_Ptr<NN_Link>{ node, 0 } };
}

/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/

NN_Test::NN_Test(const char* name) :
	NN_Layer(name)
{
}

NN_Test::NN_Test(const NN_Test& p) :
	NN_Layer(p._layer_name)
{

}

nn_shape NN_Test::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	return *input_shape[0];
}

void NN_Test::build(std::vector<nn_shape*>& input_shape) {

}

NN_Tensor<nn_type> NN_Test::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input) {
	return *input[0];
}

NN_Tensor<nn_type> NN_Test::run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output) {
	return *d_output[0];
}

/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

NN_Dense::NN_Dense(const int amounts, const char* name) :
	NN_Layer(name),
	_amounts(amounts),
	_weight(),
	_bias()
{
}

nn_shape NN_Dense::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/

	return nn_shape({ in_shape[0], _amounts });
}

void NN_Dense::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	int c = 1;
	for (int i = 1; i < in_shape.size(); ++i) c *= in_shape[i];

	_weight.set({ c, _amounts });
	set_uniform(_weight);
	_bias = NN_Tensor<nn_type>::zeros({ _amounts });
}

NN_Tensor<nn_type> NN_Dense::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input) {
	NN_Tensor<nn_type>& _input = *input[0];
	NN_Tensor<nn_type> output({ _input._shape[0], _weight._shape[1] });

	CudaTensor m_input(
		_input._data,
		_input._shape[0],
		0,
		0,
		_weight._shape[0]
	);
	CudaTensor m_weight(
		_weight._data,
		_weight._shape[0],
		0,
		0,
		_weight._shape[1]
	);
	CudaTensor m_bias(
		_bias._data,
		0,
		0,
		0,
		_bias._shape[0]
	);
	CudaTensor m_output(
		output._data,
		output._shape[0],
		0,
		0,
		output._shape[1]
	);

	dens(s, m_input, m_weight, m_output);
	add_bias(s, m_output, m_bias, m_output);

	return output;
}

NN_Tensor<nn_type> NN_Dense::run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output) {
	return *d_output[0];
}

/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

NN_ReLU::NN_ReLU(const char* name) :
	NN_Layer(name)
{
}

nn_shape NN_ReLU::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	return *input_shape[0];
}

void NN_ReLU::build(std::vector<nn_shape*>& input_shape) {

}

NN_Tensor<nn_type> NN_ReLU::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input) {
	float* data = input[0]->_data;
	NN_Tensor<nn_type> output(input[0]->_shape);

	relu(s, data, output._data, output._len);

	return output;
}

NN_Tensor<nn_type> NN_ReLU::run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output) {
	return NN_Tensor<nn_type>();
}

#endif

#ifndef FIX_MODE

/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

NN_Input::NN_Input(const nn_shape& input_size, const char* _layer_name) :
	NN_Layer(_layer_name)
{
	_shape = input_size;
	_shape.insert(_shape.begin(), -1);
}

NN_Input::~NN_Input() {

}

nn_shape NN_Input::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	if (input_shape.size() != 1) {
		ErrorExcept(
			"[NN_Input::calculate_output_size()] input node can only one input. but %d recieved.",
			input_shape.size()
		);
	}

	return _shape;
}

void NN_Input::build(std::vector<nn_shape*>& input_shape) {

}

NN_Tensor NN_Input::run_forward(cudaStream_t s, std::vector<NN_Tensor*>& input) {
	NN_Tensor output(device_t::GPU);

	input[0]->copy_to(output);

	return output;
}

NN_Tensor NN_Input::run_backward(cudaStream_t s, std::vector<NN_Tensor*>& d_output) {
	return NN_Tensor(device_t::GPU);
}


Layer_t Input(const std::vector<int>& input_size, const char* layer_name) {
	NN_Layer* layer = new NN_Input(input_size, layer_name);
	NN_Link* node = new NN_Link;

	node->_forward = layer;

	NN_Manager::add_node(node);
	NN_Manager::add_layer(layer);

	return { Layer_Ptr<NN_Link>{ node, 0 } };
}

/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/
/*
NN_Test::NN_Test(const char* name) :
	NN_Layer(name)
{
}

NN_Test::NN_Test(const NN_Test& p) :
	NN_Layer(p._layer_name)
{

}

nn_shape NN_Test::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	return input_shape;
}

void NN_Test::build(std::vector<nn_shape*>& input_shape) {

}

NN_Tensor NN_Test::run_forward(cudaStream_t s, std::vector<NN_Tensor*>& input) {
	NN_Tensor output;

	for (NN_Tensor* p_input : input) output.test_value += p_input->test_value;

	return output;
}

NN_Tensor NN_Test::run_backward(cudaStream_t s, std::vector<NN_Tensor*>& d_output) {
	return NN_Tensor();
}
*/
/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

NN_Dense::NN_Dense(const int amounts, const char* name) :
	NN_Layer(name),
	_amounts(amounts),
	_weight(device_t::GPU),
	_bias(device_t::GPU)
{
}

nn_shape NN_Dense::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/

	return nn_shape({ in_shape[0], _amounts });
}

void NN_Dense::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	int c = 1;
	for (int i = 1; i < in_shape.size(); ++i) c *= in_shape[i];

	_weight.set(nn_shape({ c, _amounts }));
	set_uniform(_weight);
	_bias = NN_Tensor::zeros(nn_shape({ _amounts }), device_t::GPU);
}

NN_Tensor NN_Dense::run_forward(cudaStream_t s, std::vector<NN_Tensor*>& input) {
	NN_Tensor& _input = *input[0];
	nn_shape out_shape({ _input._shape[0], _weight._shape[1] });
	NN_Tensor output(out_shape, device_t::GPU);

	NN_Tensor4D m_input(
		_input._data,
		_input._shape[0],
		0,
		0,
		_weight._shape[0]
	);
	NN_Tensor4D m_weight(
		_weight._data,
		_weight._shape[0],
		0,
		0,
		_weight._shape[1]
	);
	NN_Tensor4D m_bias(
		_bias._data,
		0,
		0,
		0,
		_bias._shape[0]
	);
	NN_Tensor4D m_output(
		output._data,
		output._shape[0],
		0,
		0,
		output._shape[1]
	);

	dens(s, m_input, m_weight, m_output);
	add_bias(s, m_output, m_bias, m_output);

	return output;
}

NN_Tensor NN_Dense::run_backward(cudaStream_t s, std::vector<NN_Tensor*>& d_output) {
	return *d_output[0];
}


#endif