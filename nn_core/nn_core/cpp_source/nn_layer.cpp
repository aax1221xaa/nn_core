#include "nn_layer.h"
#include "../cuda_source/cuda_indice.cuh"
#include "../cuda_source/cuda_misc.cuh"
#include "../cuda_source/dens.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/convolution.cuh"
#include "../cuda_source/maxpool.cuh"


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

		for (nn_shape::iterator i = _shape.begin(), j = input_shape[0]->begin(); i != _shape.end(); ++i, ++j) {
			if (*i >= 0 && *i != *j) {
				ErrorExcept(
					"[NN_Input::calculate_output_size()] input layer expected %s. but received %s.",
					put_shape(_shape), put_shape(*input_shape[0])
				);
			}
		}

		out_shape = *input_shape[0];
	}
}

void NN_Input::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Input::set_io(nn_shape& out_shape, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	DeviceTensor<nn_type>& _input = *input[0];

	output = _input;
	output._shape = out_shape;
}

void NN_Input::run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

NN_Backward* NN_Input::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Input;
}

void NN_D_Input::set_dio(
	std::vector<nn_shape*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
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

	out_shape = { in_shape.front(), _amounts };
}

void NN_Dense::build(std::vector<nn_shape*>& input_shape) {
	nn_shape& in_shape = *input_shape[0];

	_weight.set({ in_shape[1], _amounts });
	set_uniform(_weight);
	_bias = DeviceTensor<nn_type>::zeros({ _amounts });
}

void NN_Dense::run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	DeviceTensor<nn_type>& _input = *input[0];

	Tensor4D m_input = {
		_input._data,
		_input._shape[0],
		_input._shape[1],
		1, 1
	};
	Tensor4D m_weight = {
		_weight._data,
		_weight._shape[0],
		_weight._shape[1],
		1, 1
	};
	Tensor4D m_bias = {
		_bias._data,
		_bias._shape[0],
		1, 1, 1
	};
	Tensor4D m_output = {
		output._data,
		output._shape[0],
		output._shape[1],
		1, 1
	};

	dense(m_input, m_weight, m_output);
	add_bias_1d(m_output, m_bias, m_output);
}

NN_Backward* NN_Dense::create_backward(NN_Optimizer& optimizer) {
	NN_Optimizer* _optimizer = optimizer.create();

	return new NN_D_Dense(_optimizer, this);
}

NN_D_Dense::NN_D_Dense(NN_Optimizer* optimizer, NN_Dense* layer) :
	_optimizer(optimizer),
	_weight(layer->_weight),
	_bias(layer->_bias)
{
	_w_grad = DeviceTensor<nn_type>::zeros_like(_weight);
	_b_grad = DeviceTensor<nn_type>::zeros_like(_bias);

	_t_weight.set({ _weight._shape[1], _weight._shape[0] });

	_w_grad = DeviceTensor<nn_type>::zeros_like(_weight);
	_b_grad = DeviceTensor<nn_type>::zeros_like(_bias);

	_optimizer->set(_weight, _bias);
}

NN_D_Dense::~NN_D_Dense() {
	delete _optimizer;
}

void NN_D_Dense::set_dio(
	std::vector<Dimension*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
) {
	Dimension& shape = *in_shape[0];
	
	_t_input.set({ shape[1], shape[0] });
	d_inputs[0]->set(shape);
}

void NN_D_Dense::run_backward(
	cudaStream_t* s,
	std::vector<DeviceTensor<nn_type>*>& inputs,
	DeviceTensor<nn_type>& outputs,
	DeviceTensor<nn_type>& d_output,
	std::vector<DeviceTensor<nn_type>*>& d_input
) {
	Tensor4D m_doutput = {
		d_output._data,
		d_output._shape[0],
		d_output._shape[1],
		1, 1
	};
	Tensor4D m_weight = {
		_weight._data,
		_weight._shape[0],
		_weight._shape[1],
		1, 1
	};
	Tensor4D m_tweight = {
		_t_weight._data,
		_t_weight._shape[0],
		_t_weight._shape[1],
		1, 1
	};
	Tensor4D m_dinput = {
		d_input[0]->_data,
		d_input[0]->_shape[0],
		d_input[0]->_shape[1],
		1, 1
	};

	transpose(m_weight, m_tweight);
	dense(m_doutput, m_tweight, m_dinput);

	Tensor4D m_input = {
		inputs[0]->_data,
		inputs[0]->_shape[0],
		inputs[0]->_shape[1],
		1, 1
	};
	Tensor4D m_tinput = {
		_t_input._data,
		_t_input._shape[0],
		_t_input._shape[1],
		1, 1
	};
	Tensor4D m_wgradient = {
		_w_grad._data,
		_w_grad._shape[0],
		_w_grad._shape[1],
		1, 1,
	};

	transpose(m_input, m_tinput);
	dense(m_doutput, m_tinput, m_wgradient);

	Tensor4D m_bgradient = {
		_b_grad._data,
		_b_grad._shape[0],
		1, 1, 1
	};

	sum_gradient_1d(m_doutput, m_bgradient);

	_optimizer->run(_w_grad, _b_grad);
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

void NN_ReLU::calculate_output_size(std::vector<Dimension*>& input_shape, Dimension& out_shape) {
	out_shape = *input_shape[0];
}

void NN_ReLU::build(std::vector<Dimension*>& input_shape) {

}

void NN_ReLU::run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	Tensor4D m_input, m_output;

	if (input[0]->_shape.get_ranks() == 2) {
		m_input._data = input[0]->_data;
		m_input._n = input[0]->_shape[0];
		m_input._c = input[0]->_shape[1];
		m_input._h = m_input._w = 1;

		m_output._data = output._data;
		m_output._n = output._shape[0];
		m_output._c = output._shape[1];
		m_output._h = m_output._w = 1;
	}
	else {
		m_input._data = input[0]->_data;
		m_input._n = input[0]->_shape[0];
		m_input._c = input[0]->_shape[1];
		m_input._h = input[0]->_shape[2];
		m_input._w = input[0]->_shape[3];

		m_output._data = output._data;
		m_output._n = output._shape[0];
		m_output._c = output._shape[1];
		m_output._h = output._shape[2];
		m_output._w = output._shape[3];
	}

	relu(m_input, m_output);
}

NN_Backward* NN_ReLU::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_ReLU();
}

void NN_D_ReLU::set_dio(
	std::vector<Dimension*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
) {
	Dimension& shape = *in_shape[0];

	d_inputs[0]->set(shape);
}

void NN_D_ReLU::run_backward(
	cudaStream_t* s,
	std::vector<DeviceTensor<nn_type>*>& inputs,
	DeviceTensor<nn_type>& outputs,
	DeviceTensor<nn_type>& d_output,
	std::vector<DeviceTensor<nn_type>*>& d_input
) {
	Tensor4D m_doutput, m_dinput, m_input;

	m_doutput._data = d_output._data;
	m_doutput._n = d_output._shape[0];
	m_doutput._c = d_output._shape[1];

	m_input._data = inputs[0]->_data;
	m_input._n = inputs[0]->_shape[0];
	m_input._c = inputs[0]->_shape[1];

	m_dinput._data = d_input[0]->_data;
	m_dinput._n = d_input[0]->_shape[0];
	m_dinput._c = d_input[0]->_shape[1];

	if (d_output._shape.get_ranks() == 2) {
		m_doutput._h = m_doutput._w = 1;
		m_dinput._h = m_dinput._w = 1;
		m_input._h = m_input._w = 1;
	}
	else {
		m_doutput._h = d_output._shape[2];
		m_doutput._w = d_output._shape[3];

		m_dinput._h = d_input[0]->_shape[2];
		m_dinput._w = d_input[0]->_shape[3];

		m_input._h = inputs[0]->_shape[2];
		m_input._w = inputs[0]->_shape[3];
	}

	d_relu(m_doutput, m_input, m_dinput);
}

/**********************************************/
/*                                            */
/*                  NN_Conv2D                 */
/*                                            */
/**********************************************/

NN_Conv2D::NN_Conv2D(int amounts, const Dimension& kernel_size, const Dimension& strides, Pad pad, const char* name) :
	NN_Layer(name),
	_amounts(amounts),
	_kernel_size(kernel_size),
	_strides(strides),
	_pad(pad),
	_do_padding(false)
{
}

void NN_Conv2D::calculate_output_size(std::vector<Dimension*>& input_shape, Dimension& out_shape) {
	Dimension& shape = *input_shape[0];

	//printf("dim= %s\n", dimension_to_str(shape));

	if (shape.get_ranks() != 4) {
		ErrorExcept(
			"[NN_Conv2D::calculate_output_size()] invalid input shape %s.",
			shape.to_string()
		);
	}
	for (int i = 1; i < shape.get_ranks(); ++i) {
		if (shape[i] < 1) {
			ErrorExcept(
				"[NN_Conv2D::calculate_output_size()] invalid input shape %s.",
				shape.to_string()
			);
		}
	}

	if (_pad == Pad::SAME) {
		_do_padding = true;

		int n = __min(STREAMS, shape[0]);
		int c = shape[1];
		int h = (shape[2] - 1) * _strides[1] + _kernel_size[1];
		int w = (shape[3] - 1) * _strides[0] + _kernel_size[0];

		pad = DeviceTensor<nn_type>::zeros({ n, c, h, w });

		out_shape = { shape[0], _amounts, shape[2], shape[3] };
	}
	else {
		int out_h = (int)ceil((float)(shape[2] - _kernel_size[0]) / _strides[0]) + 1;
		int out_w = (int)ceil((float)(shape[3] - _kernel_size[1]) / _strides[1]) + 1;

		int n = __min(STREAMS, shape[0]);
		int c = shape[1];
		int h = (out_h - 1) * _strides[1] + _kernel_size[1];
		int w = (out_w - 1) * _strides[0] + _kernel_size[0];

		if (shape[2] != h || shape[3] != w) {
			//printf("pad= %d, %d, %d, %d\n", n, c, h, w);
			_do_padding = true;
			pad = DeviceTensor<nn_type>::zeros({ n, c, h, w });
		}
		out_shape = { shape[0], _amounts, out_h, out_w };
	}
}

void NN_Conv2D::build(std::vector<Dimension*>& input_shape) {
	Dimension& shape = *input_shape[0];

	_kernel.set({ _amounts, shape[1], _kernel_size[1], _kernel_size[0] });
	_bias = DeviceTensor<nn_type>::zeros({ _amounts });

	set_uniform(_kernel);
}

void NN_Conv2D::run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	Tensor4D m_input = {
		NULL,
		input[0]->_shape[0],
		input[0]->_shape[1],
		input[0]->_shape[2],
		input[0]->_shape[3]
	};
	Tensor4D m_kernel = {
		_kernel._data,
		_kernel._shape[0],
		_kernel._shape[1],
		_kernel._shape[2],
		_kernel._shape[3]
	};
	Tensor4D m_bias = {
		_bias._data,
		_bias._shape[0],
		1, 1, 1
	};
	Tensor4D m_output = {
		NULL,
		output._shape[0],
		output._shape[1],
		output._shape[2],
		output._shape[3]
	};
	Tensor4D m_pad = {
		NULL,
		pad._shape[0],
		pad._shape[1],
		pad._shape[2],
		pad._shape[3]
	};

	uint *indice = new uint[m_kernel._c * m_kernel._h * m_kernel._w];

	for (uint c = 0; c < m_kernel._c; ++c) {
		for (uint h = 0; h < m_kernel._h; ++h) {
			for (uint w = 0; w < m_kernel._w; ++w) {
				indice[c * (m_kernel._h * m_kernel._w) + h * m_kernel._w + w] =
					_do_padding ?
					c * (m_pad._h * m_pad._w) + h * m_pad._w + w :
					c * (m_input._h * m_input._w) + h * m_input._w + w;
			}
		}
	}

	set_indice(indice, sizeof(uint) * m_kernel._c * m_kernel._h * m_kernel._w, 0);
	cuint* const_indice = get_indice_ptr();

	delete[] indice;

	for (uint i = 0; i < m_input._n; ++i) {
		m_input._data = input[0]->_data + (i * m_input._c * m_input._h * m_input._w);
		m_output._data = output._data + (i * m_output._c * m_output._h * m_output._w);

		if (_do_padding) {
			m_pad._data = pad._data + ((i % STREAMS) * m_pad._h * m_pad._w);

			padding_dilation(
				s[i % STREAMS], 
				m_input, m_pad, 
				(m_pad._w - m_input._w) / 2, 
				(m_pad._h - m_input._h) / 2, 
				1, 1
			);
			conv2d(
				s[i % STREAMS],
				const_indice,
				m_pad,
				m_kernel,
				m_output,
				_strides[1],
				_strides[0]
			);
		}
		else {
			conv2d(
				s[i % STREAMS],
				const_indice,
				m_input,
				m_kernel,
				m_output,
				_strides[1],
				_strides[0]
			);
		}
	}

	//printf("================================\n");
	//printf("layer name= %s\n", _layer_name);
	//printf("d_input= %s\n", dimension_to_str(m_input._shape));
	//printf("d_kernel= %s\n", dimension_to_str(k_shape));
	//printf("d_output = %s\n", dimension_to_str(out_shape));
}

NN_Backward* NN_Conv2D::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Conv2D(optimizer.create(), this);
}

NN_D_Conv2D::NN_D_Conv2D(NN_Optimizer* optimizer, NN_Conv2D* layer) :
	_optimizer(optimizer),
	_kernel(layer->_kernel),
	_bias(layer->_bias),
	_strides(layer->_strides)
{
	_optimizer->set(_kernel, _bias);

	_w_grad = DeviceTensor<nn_type>::zeros_like(_kernel);
	_b_grad = DeviceTensor<nn_type>::zeros_like(_bias);
}

NN_D_Conv2D::~NN_D_Conv2D() {
	delete _optimizer;
}

void NN_D_Conv2D::set_dio(
	std::vector<Dimension*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
) {
	Dimension& in_size = *in_shape[0];
	Dimension& k_size = _kernel._shape;
	Dimension& out_size = d_outputs[0]->_shape;

	d_inputs[0]->set(in_size);
	_t_kernel.set({ k_size[1], k_size[0], k_size[2], k_size[3] });

	int pad_n = __min(STREAMS, out_size[0]);
	int pad_c = out_size[1];
	int pad_h = (out_size[2] * _strides[1]) + k_size[2] - 1;
	int pad_w = (out_size[3] * _strides[0]) + k_size[3] - 1;

	_pad = DeviceTensor<nn_type>::zeros({ pad_n, pad_c, pad_h, pad_w });
}

void NN_D_Conv2D::run_backward(
	cudaStream_t* s,
	std::vector<DeviceTensor<nn_type>*>& inputs,
	DeviceTensor<nn_type>& outputs,
	DeviceTensor<nn_type>& d_output,
	std::vector<DeviceTensor<nn_type>*>& d_input
) {
	Tensor4D m_input = inputs[0]->get_tensor4d(0, -1);
	Tensor4D doutput = d_output.get_tensor4d(0, -1);
	Tensor4D dpad = _pad.get_tensor4d(0, -1);
	Tensor4D kernel = _kernel.get_tensor4d(0, -1);
	Tensor4D t_kernel = _t_kernel.get_tensor4d(0, -1);
	Tensor4D dinput = d_input[0]->get_tensor4d(0, -1);
	Tensor4D bias = _bias.get_tensor4d(0, -1);

	uint* indice = new uint[t_kernel._c * t_kernel._h * t_kernel._w];

	for (uint c = 0; c < t_kernel._c; ++c) {
		for (uint h = 0, _h = t_kernel._h - 1; h < t_kernel._h; ++h, --_h) {
			for (uint w = 0, _w = t_kernel._w - 1; w < t_kernel._w; ++w, --_w) {
				indice[c * (t_kernel._h * t_kernel._w) + h * t_kernel._w + w] =
					c * (dpad._h * dpad._w) + _h * dpad._w + _w;
			}
		}
	}

	set_indice(indice, sizeof(uint) * t_kernel._c * t_kernel._h * t_kernel._w, 0);
	uint* const_indice = get_indice_ptr();

	delete[] indice;

	transpose(kernel, t_kernel);

	for (uint i = 0; i < doutput._n; ++i) {
		doutput._data = d_output._data + (i * doutput._c * doutput._h * doutput._w);
		dpad._data = _pad._data + ((i % STREAMS) * dpad._h * dpad._w);
		dinput._data = d_input[0]->_data + (i * dinput._c * dinput._h * dinput._w);

		padding_dilation(
			s[i % STREAMS],
			doutput,
			dpad,
			(dpad._w - doutput._w) / 2,
			(dpad._h - doutput._h) / 2,
			_strides[0],
			_strides[1]
		);
		conv2d(
			s[i % STREAMS],
			const_indice,
			dpad,
			t_kernel,
			dinput,
			1, 1
		);
	}

	doutput = d_output.get_tensor4d(0, -1);
	kernel_conv2d(
		doutput,
		m_input,
		_w_grad.get_tensor4d(0, -1)
	);
	sum_gradient_2d(
		m_input,
		_b_grad.get_tensor4d(0, -1)
	);

	_optimizer->run(_w_grad, _b_grad);
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

void NN_Flatten::calculate_output_size(std::vector<Dimension*>& input_shape, Dimension& out_shape) {
	int c = 1;
	Dimension& in_shape = *input_shape[0];

	for (int i = 1; i < in_shape.get_ranks(); ++i) c *= in_shape[i];

	out_shape = { in_shape[0], c };
}

void NN_Flatten::build(std::vector<Dimension*>& input_shape) {

}

void NN_Flatten::set_io(Dimension& out_shape, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	DeviceTensor<nn_type>& _input = *input[0];

	output = _input;
	output._shape = out_shape;
}

void NN_Flatten::run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

NN_Backward* NN_Flatten::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Flatten;
}

void NN_D_Flatten::set_dio(
	std::vector<Dimension*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
) {
	*d_inputs[0] = *d_outputs[0];
	d_inputs[0]->_shape = *in_shape[0];
}

void NN_D_Flatten::run_backward(
	cudaStream_t* s,
	std::vector<DeviceTensor<nn_type>*>& inputs,
	DeviceTensor<nn_type>& outputs,
	DeviceTensor<nn_type>& d_output,
	std::vector<DeviceTensor<nn_type>*>& d_input
) {

}

/**********************************************/
/*                                            */
/*                 NN_Maxpool2D               */
/*                                            */
/**********************************************/

NN_Maxpool2D::NN_Maxpool2D(const Dimension& kernel_size, const Dimension& strides, Pad pad, const char* name) :
	_kernel_size(kernel_size),
	_strides(strides),
	_pad(pad),
	_indice(NULL),
	NN_Layer(name)
{
}

NN_Maxpool2D::~NN_Maxpool2D() {
	cudaFree(_indice);
}

void NN_Maxpool2D::calculate_output_size(std::vector<Dimension*>& input_shape, Dimension& out_shape) {
	Dimension& shape = *input_shape[0];

	int out_n = shape[0];
	int out_c = shape[1];
	int out_h = 0;
	int out_w = 0;

	if (_pad == Pad::VALID) {
		out_h = (shape[2] - _kernel_size[1]) / _strides[1] + 1;
		out_w = (shape[3] - _kernel_size[0]) / _strides[0] + 1;
	}
	else {
		out_h = (int)ceilf(float(shape[2] - _kernel_size[1]) / _strides[1] + 1);
		out_w = (int)ceilf(float(shape[3] - _kernel_size[0]) / _strides[0] + 1);

		int pad_n = __min(STREAMS, shape[0]);
		int pad_c = shape[1];
		int pad_h = (out_h - 1) * _strides[1] + _kernel_size[1];
		int pad_w = (out_w - 1) * _strides[0] + _kernel_size[0];

		if (pad_h != shape[2] || pad_w != shape[3]) {
			pad_input.set({ pad_n, pad_c, pad_h, pad_w });
		}
	}

	int h = (shape[2] - _kernel_size[1]) / _strides[1] + 1;
	int w = (shape[3] - _kernel_size[0]) / _strides[0] + 1;

	out_shape = { shape[0], shape[1], h, w };
	check_cuda(cudaMalloc(&_indice, sizeof(uint) * out_shape.get_len()));
}

void NN_Maxpool2D::build(std::vector<Dimension*>& input_shape) {

}

void NN_Maxpool2D::run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	int tile_w = calc_tile_size(_kernel_size[0], _strides[0]);
	int tile_h = calc_tile_size(_kernel_size[1], _strides[1]);

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, output._shape[3], output._shape[2]);

	Tensor4D m_input = input[0]->get_tensor4d(0, -1);
	Tensor4D m_output = output.get_tensor4d(0, -1);

	if (_pad == Pad::VALID) {
		for (uint i = 0; i < m_input._n; ++i) {
			m_input._data = input[0]->_data + (i * m_input._c * m_input._h * m_input._w);
			m_output._data = output._data + (i * m_output._c * m_output._h * m_output._w);
			uint* m_indice = _indice + (i * m_output._c * m_output._h * m_output._w);

			maxpool2d(
				s[i % STREAMS],
				m_input,
				m_output,
				m_indice,
				_kernel_size[1],
				_kernel_size[0],
				_strides[1],
				_strides[0],
				tile_h,
				tile_w
			);
		}
	}
	else {
		Tensor4D m_pad = pad_input.get_tensor4d(0, -1);

		for (uint i = 0; i < m_input._n; ++i) {
			m_input._data = input[0]->_data + (i * m_input._c * m_input._h * m_input._w);
			m_pad._data = pad_input._data + ((i % STREAMS) * m_pad._c * m_pad._h * m_pad._w);
			m_output._data = output._data + (i * m_output._c * m_output._h * m_output._w);
			uint* m_indice = _indice + (i * m_output._c * m_output._h * m_output._w);

			padding_dilation(
				s[i % STREAMS],
				m_input,
				m_pad,
				0, 0,
				1, 1
			);
			maxpool2d(
				s[i % STREAMS],
				m_pad,
				m_output,
				m_indice,
				_kernel_size[1],
				_kernel_size[0],
				_strides[1],
				_strides[0],
				tile_h,
				tile_w
			);
		}
	}
}

NN_Backward* NN_Maxpool2D::create_backward(NN_Optimizer& optimizer) {
	return new NN_D_Maxpool2D(*this);
}

int NN_Maxpool2D::calc_tile_size(int k_size, int stride) {
	return (BLOCK_32 - 1) * stride + k_size;
}

NN_D_Maxpool2D::NN_D_Maxpool2D(const NN_Maxpool2D& p) :
	_indice(p._indice),
	_kernel_size(p._kernel_size),
	_strides(p._strides)
{
}

void NN_D_Maxpool2D::set_dio(
	std::vector<Dimension*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
) {
	d_inputs[0]->set(*in_shape[0]);
}

void NN_D_Maxpool2D::run_backward(
	cudaStream_t* s,
	std::vector<DeviceTensor<nn_type>*>& inputs,
	DeviceTensor<nn_type>& outputs,
	DeviceTensor<nn_type>& d_output,
	std::vector<DeviceTensor<nn_type>*>& d_input
) {
	d_maxpool2d(
		s,
		d_output.get_tensor4d(0, -1),
		_indice,
		d_input[0]->get_tensor4d(0, -1),
		_kernel_size[0],
		_strides[1],
		_strides[0]
	);
}

/**********************************************/
/*                                            */
/*                  NN_SoftMax                */
/*                                            */
/**********************************************/
