#include "nn_layer.h"
#include "../cuda_source/dens.cuh"
#include "../cuda_source/add_bias.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/convolution.cuh"
#include "../cuda_source/maxpool.cuh"


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

void NN_Input::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	if (input_shape.size() == 0) out_shape = _shape;
	else {
		if (input_shape.size() > 1) {
			ErrorExcept(
				"[NN_Input::calculate_output_size()] input layer can't receive %d layers.",
				input_shape.size()
			);
		}
		else if (input_shape[0]->size() != _shape.size()) {
			ErrorExcept(
				"[NN_Input::calculate_output_size()] input layer expected %ld dimensions. but received %ld dimensions.",
				_shape.size(), input_shape[0]->size()
			);
		}

		for (int i = 0; i < _shape.size(); ++i) {
			if (_shape[i] >= 0 && _shape[i] != (*input_shape[0])[i]) {
				ErrorExcept(
					"[NN_Input::calculate_output_size()] input layer expected %s. but received %s.",
					dimension_to_str(_shape), dimension_to_str(*input_shape[0])
				);
			}
		}
		out_shape = *input_shape[0];
	}
}

void NN_Input::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Input::set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& _input = *input[0];

	output = _input;
	output._shape = out_shape;
}

void NN_Input::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

NN_Backward* NN_Input::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Input;
}

void NN_D_Input::set_dio(
	std::vector<nn_shape*>& in_shape,
	std::vector<NN_Tensor<nn_type>*>& d_outputs,
	std::vector<NN_Tensor<nn_type>*>& d_inputs
) {
	*d_inputs[0] = *d_outputs[0];
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
/*
NN_Test::NN_Test(const char* name) :
	NN_Layer(name)
{
}

NN_Test::NN_Test(const NN_Test& p) :
	NN_Layer(p._layer_name)
{

}

void NN_Test::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	*input_shape[0] = out_shape;
}

void NN_Test::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Test::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	*input[0] = output;
}

void NN_Test::run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {
	//return *d_output[0];
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
	_weight(),
	_bias()
{
}

void NN_Dense::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	nn_shape& in_shape = *input_shape[0];

	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/

	out_shape = nn_shape({ in_shape[0], _amounts });
}

void NN_Dense::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	int c = 1;
	for (int i = 1; i < in_shape.size(); ++i) c *= in_shape[i];

	_weight.set({ c, _amounts });
	set_uniform(_weight);
	_bias = NN_Tensor<nn_type>::zeros({ _amounts });
}

void NN_Dense::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& _input = *input[0];
	//NN_Tensor<nn_type> output({ _input._shape[0], _weight._shape[1] });

	CudaTensor m_input(
		_input._data,
		_input._shape[0],
		_weight._shape[0],
		1,
		1
	);
	CudaTensor m_weight(
		_weight._data,
		_weight._shape[0],
		_weight._shape[1],
		1,
		1
	);
	CudaTensor m_bias(
		_bias._data,
		1,
		_bias._shape[0],
		1,
		1
	);
	CudaTensor m_output(
		output._data,
		output._shape[0],
		output._shape[1],
		1,
		1
	);

	dense(NULL, m_input, m_weight, m_output);
	add_bias(NULL, m_output, m_bias, m_output);
}

NN_Backward* NN_Dense::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Dense(optimizer, this);
}

NN_D_Dense::NN_D_Dense(NN_Optimizer& optimizer, NN_Dense* layer) :
	_optimizer(optimizer),
	_weight(layer->_weight),
	_bias(layer->_bias)
{
	_w_grad = NN_Tensor<nn_type>::zeros_like(_weight);
	_b_grad = NN_Tensor<nn_type>::zeros_like(_bias);
}

void NN_D_Dense::set_dio(
	std::vector<nn_shape*>& in_shape,
	std::vector<NN_Tensor<nn_type>*>& d_outputs,
	std::vector<NN_Tensor<nn_type>*>& d_inputs
) {
	nn_shape& shape = *in_shape[0];

	d_inputs[0]->set(shape);
}

void NN_D_Dense::run_backward(
	cudaStream_t* s,
	std::vector<NN_Tensor<nn_type>*>& inputs,
	NN_Tensor<nn_type>& outputs,
	NN_Tensor<nn_type>& d_output,
	std::vector<NN_Tensor<nn_type>*>& d_input
) {

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

void NN_ReLU::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	out_shape = *input_shape[0];
}

void NN_ReLU::build(std::vector<nn_shape*>& input_shape) {

}

void NN_ReLU::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	float* data = input[0]->_data;

	relu(NULL, data, output._data, output._len);
}

NN_Backward* NN_ReLU::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_ReLU();
}

void NN_D_ReLU::set_dio(
	std::vector<nn_shape*>& in_shape,
	std::vector<NN_Tensor<nn_type>*>& d_outputs,
	std::vector<NN_Tensor<nn_type>*>& d_inputs
) {
	nn_shape& shape = *in_shape[0];

	d_inputs[0]->set(shape);
}

void NN_D_ReLU::run_backward(
	cudaStream_t* s,
	std::vector<NN_Tensor<nn_type>*>& inputs,
	NN_Tensor<nn_type>& outputs,
	NN_Tensor<nn_type>& d_output,
	std::vector<NN_Tensor<nn_type>*>& d_input
) {

}

/**********************************************/
/*                                            */
/*                  NN_Conv2D                 */
/*                                            */
/**********************************************/

size_t NN_Conv2D::const_offset_cnt = 0;

NN_Conv2D::NN_Conv2D(int amounts, const nn_shape& kernel_size, const nn_shape& strides, Pad pad, const char* name) :
	NN_Layer(name),
	_amounts(amounts),
	_kernel_size(kernel_size),
	_strides(strides),
	_pad(pad),
	_do_padding(false),
	const_offset(0)
{
}

void NN_Conv2D::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	nn_shape& shape = *input_shape[0];

	//printf("dim= %s\n", dimension_to_str(shape));

	if (shape.size() != 4) {
		ErrorExcept(
			"[NN_Conv2D::calculate_output_size()] invalid input shape %s.",
			dimension_to_str(shape)
		);
	}
	for (int i = 1; i < shape.size(); ++i) {
		if (shape[i] < 1) {
			ErrorExcept(
				"[NN_Conv2D::calculate_output_size()] invalid input shape %s.",
				dimension_to_str(shape)
			);
		}
	}

	if (_pad == Pad::SAME) {
		_do_padding = true;

		int n = STREAMS;
		int c = shape[1];
		int h = (shape[2] - 1) * _strides[1] + _kernel_size[1];
		int w = (shape[3] - 1) * _strides[0] + _kernel_size[0];

		pad = NN_Tensor<nn_type>::zeros({ n, c, h, w });

		out_shape = { shape[0], _amounts, shape[2], shape[3] };
	}
	else {
		int out_h = (int)ceil((float)(shape[2] - _kernel_size[0]) / _strides[0]) + 1;
		int out_w = (int)ceil((float)(shape[3] - _kernel_size[1]) / _strides[1]) + 1;

		int n = STREAMS;
		int c = shape[1];
		int h = (out_h - 1) * _strides[1] + _kernel_size[1];
		int w = (out_w - 1) * _strides[0] + _kernel_size[0];

		if (shape[2] != h || shape[3] != w) {
			//printf("pad= %d, %d, %d, %d\n", n, c, h, w);
			_do_padding = true;
			pad = NN_Tensor<nn_type>::zeros({ n, c, h, w });
		}
		out_shape = { shape[0], _amounts, out_h, out_w };
	}
}

void NN_Conv2D::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& shape = *input_shape[0];

	_kernel.set({ _amounts, shape[1], _kernel_size[1], _kernel_size[0] });
	_bias = NN_Tensor<nn_type>::zeros({ _amounts });

	set_uniform(_kernel);

	size_t indice_size = (size_t)(_kernel_size[1] * _kernel_size[0]);
	uint* indice = new uint[indice_size];

	for (int y = 0; y < _kernel_size[1]; ++y) {
		for (int x = 0; x < _kernel_size[0]; ++x) {
			indice[y * _kernel_size[0] + x] = uint(y * shape[3] + x);
		}
	}

	copy_to_indice(indice, sizeof(uint) * indice_size, sizeof(uint) * const_offset_cnt);
	const_offset = const_offset_cnt;
	const_offset_cnt += indice_size;

	delete[] indice;
}

void NN_Conv2D::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& m_input = *input[0];
	
	float* in_data = m_input._data;
	nn_shape& in_shape = m_input._shape;

	float* k_data = _kernel._data;
	nn_shape& k_shape = _kernel._shape;

	float* out_data = output._data;
	nn_shape& out_shape = output._shape;

	CudaTensor d_input(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
	CudaTensor d_kernel(k_data, k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
	CudaTensor d_output(out_data, out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

	//printf("================================\n");
	//printf("layer name= %s\n", _layer_name);
	//printf("d_input= %s\n", dimension_to_str(m_input._shape));
	//printf("d_kernel= %s\n", dimension_to_str(k_shape));
	//printf("d_output = %s\n", dimension_to_str(out_shape));

	if (_do_padding) {
		float* pad_data = pad._data;
		nn_shape& pad_shape = pad._shape;

		CudaTensor d_pad(pad_data, pad_shape[0], pad_shape[1], pad_shape[2], pad_shape[3]);

		padding_conv_2d(
			s,
			d_input,
			d_pad,
			d_kernel,
			d_output,
			_strides[0],
			_strides[1],
			const_offset
		);
	}
	else {
		conv_2d(
			s,
			d_input,
			d_kernel,
			d_output,
			_strides[0],
			_strides[1],
			const_offset
		);
	}
}

NN_Backward* NN_Conv2D::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Conv2D(optimizer, this);
}

NN_D_Conv2D::NN_D_Conv2D(NN_Optimizer& optimizer, NN_Conv2D* layer) :
	_optimizer(optimizer),
	_kernel(layer->_kernel),
	_bias(layer->_bias)
{
	_w_grad = NN_Tensor<nn_type>::zeros_like(_kernel);
	_b_grad = NN_Tensor<nn_type>::zeros_like(_bias);
}

void NN_D_Conv2D::set_dio(
	std::vector<nn_shape*>& in_shape,
	std::vector<NN_Tensor<nn_type>*>& d_outputs,
	std::vector<NN_Tensor<nn_type>*>& d_inputs
) {

}

void NN_D_Conv2D::run_backward(
	cudaStream_t* s,
	std::vector<NN_Tensor<nn_type>*>& inputs,
	NN_Tensor<nn_type>& outputs,
	NN_Tensor<nn_type>& d_output,
	std::vector<NN_Tensor<nn_type>*>& d_input
) {

}

/**********************************************/
/*                                            */
/*                  NN_Flatten                */
/*                                            */
/**********************************************/

NN_Flatten::NN_Flatten(const char* name) :
	NN_Layer(name)
{
}

void NN_Flatten::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	int c = 1;
	nn_shape& in_shape = *input_shape[0];

	for (int i = 1; i < in_shape.size(); ++i) c *= in_shape[i];

	out_shape = { in_shape[0], c };
}

void NN_Flatten::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Flatten::set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& _input = *input[0];

	output = _input;
	output._shape = out_shape;
}

void NN_Flatten::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

void NN_Flatten::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

}

/**********************************************/
/*                                            */
/*                 NN_Maxpool2D               */
/*                                            */
/**********************************************/

NN_Maxpool2D::NN_Maxpool2D(const nn_shape& kernel_size, const nn_shape& strides, const char* name) :
	_kernel_size(kernel_size),
	_strides(strides),
	NN_Layer(name)
{
}

void NN_Maxpool2D::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	nn_shape& shape = *input_shape[0];

	int h = (shape[2] - _kernel_size[1]) / _strides[1] + 1;
	int w = (shape[3] - _kernel_size[0]) / _strides[0] + 1;

	out_shape = { shape[0], shape[1], h, w };
}

void NN_Maxpool2D::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Maxpool2D::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& m_input = *input[0];

	float* in_data = m_input._data;
	nn_shape& in_shape = m_input._shape;

	float* out_data = output._data;
	nn_shape& out_shape = output._shape;

	CudaTensor d_input(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
	CudaTensor d_output(out_data, out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

	maxpool_2d(
		s,
		d_input,
		d_output,
		_kernel_size[0],
		_kernel_size[1],
		_strides[0],
		_strides[1]
	);
}

void NN_Maxpool2D::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

}

#endif

#ifndef FIX_MODE

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

void NN_Input::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	if (input_shape.size() == 0) out_shape = _shape;
	else {
		if (input_shape.size() > 1) {
			ErrorExcept(
				"[NN_Input::calculate_output_size()] input layer can't receive %d layers.",
				input_shape.size()
			);
		}
		else if (input_shape[0]->size() != _shape.size()) {
			ErrorExcept(
				"[NN_Input::calculate_output_size()] input layer expected %ld dimensions. but received %ld dimensions.",
				_shape.size(), input_shape[0]->size()
			);
		}

		for (int i = 0; i < _shape.size(); ++i) {
			if (_shape[i] >= 0 && _shape[i] != (*input_shape[0])[i]) {
				ErrorExcept(
					"[NN_Input::calculate_output_size()] input layer expected %s. but received %s.",
					dimension_to_str(_shape), dimension_to_str(*input_shape[0])
				);
			}
		}
		out_shape = *input_shape[0];
	}
}

void NN_Input::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Input::set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& _input = *input[0];

	output = _input;
	output._shape = out_shape;
}

void NN_Input::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

void NN_Input::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

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
/*
NN_Test::NN_Test(const char* name) :
	NN_Layer(name)
{
}

NN_Test::NN_Test(const NN_Test& p) :
	NN_Layer(p._layer_name)
{

}

void NN_Test::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	*input_shape[0] = out_shape;
}

void NN_Test::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Test::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	*input[0] = output;
}

void NN_Test::run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {
	//return *d_output[0];
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
	_weight(),
	_bias()
{
}

void NN_Dense::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	nn_shape& in_shape = *input_shape[0];

	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/

	out_shape = nn_shape({ in_shape[0], _amounts });
}

void NN_Dense::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	int c = 1;
	for (int i = 1; i < in_shape.size(); ++i) c *= in_shape[i];

	_weight.set({ c, _amounts });
	set_uniform(_weight);
	_bias = NN_Tensor<nn_type>::zeros({ _amounts });
}

void NN_Dense::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& _input = *input[0];
	//NN_Tensor<nn_type> output({ _input._shape[0], _weight._shape[1] });

	CudaTensor m_input(
		_input._data,
		_input._shape[0],
		_weight._shape[0],
		1,
		1
	);
	CudaTensor m_weight(
		_weight._data,
		_weight._shape[0],
		_weight._shape[1],
		1,
		1
	);
	CudaTensor m_bias(
		_bias._data,
		1,
		_bias._shape[0],
		1,
		1
	);
	CudaTensor m_output(
		output._data,
		output._shape[0],
		output._shape[1],
		1,
		1
	);

	dense(NULL, m_input, m_weight, m_output);
	add_bias(NULL, m_output, m_bias, m_output);
}

void NN_Dense::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {
	//return *d_output[0];
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

void NN_ReLU::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	out_shape = *input_shape[0];
}

void NN_ReLU::build(std::vector<nn_shape*>& input_shape) {

}

void NN_ReLU::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	float* data = input[0]->_data;

	relu(NULL, data, output._data, output._len);
}

void NN_ReLU::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

}

/**********************************************/
/*                                            */
/*                  NN_Conv2D                 */
/*                                            */
/**********************************************/

size_t NN_Conv2D::const_offset_cnt = 0;

NN_Conv2D::NN_Conv2D(int amounts, const nn_shape& kernel_size, const nn_shape& strides, Pad pad, const char* name) :
	NN_Layer(name),
	_amounts(amounts),
	_kernel_size(kernel_size),
	_strides(strides),
	_pad(pad),
	_do_padding(false),
	const_offset(0)
{
}

void NN_Conv2D::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	nn_shape& shape = *input_shape[0];

	//printf("dim= %s\n", dimension_to_str(shape));

	if (shape.size() != 4) {
		ErrorExcept(
			"[NN_Conv2D::calculate_output_size()] invalid input shape %s.",
			dimension_to_str(shape)
		);
	}
	for (int i = 1; i < shape.size(); ++i) {
		if (shape[i] < 1) {
			ErrorExcept(
				"[NN_Conv2D::calculate_output_size()] invalid input shape %s.",
				dimension_to_str(shape)
			);
		}
	}

	if (_pad == Pad::SAME) {
		_do_padding = true;

		int n = STREAMS;
		int c = shape[1];
		int h = (shape[2] - 1) * _strides[1] + _kernel_size[1];
		int w = (shape[3] - 1) * _strides[0] + _kernel_size[0];

		pad = NN_Tensor<nn_type>::zeros({ n, c, h, w });

		out_shape = { shape[0], _amounts, shape[2], shape[3] };
	}
	else {
		int out_h = (int)ceil((float)(shape[2] - _kernel_size[0]) / _strides[0]) + 1;
		int out_w = (int)ceil((float)(shape[3] - _kernel_size[1]) / _strides[1]) + 1;

		int n = STREAMS;
		int c = shape[1];
		int h = (out_h - 1) * _strides[1] + _kernel_size[1];
		int w = (out_w - 1) * _strides[0] + _kernel_size[0];

		if (shape[2] != h || shape[3] != w) {
			//printf("pad= %d, %d, %d, %d\n", n, c, h, w);
			_do_padding = true;
			pad = NN_Tensor<nn_type>::zeros({ n, c, h, w });
		}
		out_shape = { shape[0], _amounts, out_h, out_w };
	}
}

void NN_Conv2D::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& shape = *input_shape[0];

	_kernel.set({ _amounts, shape[1], _kernel_size[1], _kernel_size[0] });
	_bias = NN_Tensor<nn_type>::zeros({ _amounts });

	set_uniform(_kernel);

	size_t indice_size = (size_t)(_kernel_size[1] * _kernel_size[0]);
	uint* indice = new uint[indice_size];

	for (int y = 0; y < _kernel_size[1]; ++y) {
		for (int x = 0; x < _kernel_size[0]; ++x) {
			indice[y * _kernel_size[0] + x] = uint(y * shape[3] + x);
		}
	}

	copy_to_indice(indice, sizeof(uint) * indice_size, sizeof(uint) * const_offset_cnt);
	const_offset = const_offset_cnt;
	const_offset_cnt += indice_size;

	delete[] indice;
}

void NN_Conv2D::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& m_input = *input[0];

	float* in_data = m_input._data;
	nn_shape& in_shape = m_input._shape;

	float* k_data = _kernel._data;
	nn_shape& k_shape = _kernel._shape;

	float* out_data = output._data;
	nn_shape& out_shape = output._shape;

	CudaTensor d_input(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
	CudaTensor d_kernel(k_data, k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
	CudaTensor d_output(out_data, out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

	//printf("================================\n");
	//printf("layer name= %s\n", _layer_name);
	//printf("d_input= %s\n", dimension_to_str(m_input._shape));
	//printf("d_kernel= %s\n", dimension_to_str(k_shape));
	//printf("d_output = %s\n", dimension_to_str(out_shape));

	if (_do_padding) {
		float* pad_data = pad._data;
		nn_shape& pad_shape = pad._shape;

		CudaTensor d_pad(pad_data, pad_shape[0], pad_shape[1], pad_shape[2], pad_shape[3]);

		padding_conv_2d(
			s,
			d_input,
			d_pad,
			d_kernel,
			d_output,
			_strides[0],
			_strides[1],
			const_offset
		);
	}
	else {
		conv_2d(
			s,
			d_input,
			d_kernel,
			d_output,
			_strides[0],
			_strides[1],
			const_offset
		);
	}
}

void NN_Conv2D::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

}

/**********************************************/
/*                                            */
/*                  NN_Flatten                */
/*                                            */
/**********************************************/

NN_Flatten::NN_Flatten(const char* name) :
	NN_Layer(name)
{
}

void NN_Flatten::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	int c = 1;
	nn_shape& in_shape = *input_shape[0];

	for (int i = 1; i < in_shape.size(); ++i) c *= in_shape[i];

	out_shape = { in_shape[0], c };
}

void NN_Flatten::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Flatten::set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& _input = *input[0];

	output = _input;
	output._shape = out_shape;
}

void NN_Flatten::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

void NN_Flatten::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

}

/**********************************************/
/*                                            */
/*                 NN_Maxpool2D               */
/*                                            */
/**********************************************/

NN_Maxpool2D::NN_Maxpool2D(const nn_shape& kernel_size, const nn_shape& strides, const char* name) :
	_kernel_size(kernel_size),
	_strides(strides),
	NN_Layer(name)
{
}

void NN_Maxpool2D::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	nn_shape& shape = *input_shape[0];

	int h = (shape[2] - _kernel_size[1]) / _strides[1] + 1;
	int w = (shape[3] - _kernel_size[0]) / _strides[0] + 1;

	out_shape = { shape[0], shape[1], h, w };
}

void NN_Maxpool2D::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Maxpool2D::run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	NN_Tensor<nn_type>& m_input = *input[0];

	float* in_data = m_input._data;
	nn_shape& in_shape = m_input._shape;

	float* out_data = output._data;
	nn_shape& out_shape = output._shape;

	CudaTensor d_input(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
	CudaTensor d_output(out_data, out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

	maxpool_2d(
		s,
		d_input,
		d_output,
		_kernel_size[0],
		_kernel_size[1],
		_strides[0],
		_strides[1]
	);
}

void NN_Maxpool2D::run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {

}


#endif