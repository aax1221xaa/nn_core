#include "../header/nn_conv.h"
#include "../header/nn_misc.h"


NN_Conv2D::NN_Conv2D(int amounts, const NN_Shape& kernel_size, const NN_Shape& strides, const std::string& pad, const std::string& name) :
	NN_Layer(name),
	_amounts(amounts),
	_kernel_size(kernel_size),
	_strides(strides),
	_pad(pad)
{

}

NN_Conv2D::~NN_Conv2D() {

}

void NN_Conv2D::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();
	NN_Shape& out_shape = output_shape[0].val();

	if (_pad == "same") {
		int n = shape[0];
		int h = (int)ceil((float)shape[1] / _strides[1]);
		int w = (int)ceil((float)shape[2] / _strides[0]);
		int c = _amounts;

		out_shape = NN_Shape({ n, h, w, c });
	}
	else {
		int n = shape[0];
		int h = (int)floorf((float)(shape[1] - _kernel_size[1]) / _strides[1] + 1);
		int w = (int)floorf((float)(shape[2] - _kernel_size[0]) / _strides[0] + 1);
		int c = _amounts;

		out_shape = NN_Shape({ n, h, w, c });
	}
}

void NN_Conv2D::build(const NN_List<NN_Shape>& input_shape, NN_List<NN_Tensor<nn_type>>& weights) {
	const NN_Shape& shape = input_shape[0].val();

	_kernel.resize(NN_Shape({ _kernel_size[1], _kernel_size[0], shape[3], _amounts }));				// h x w x in_c x out_c
	_bias = NN_Tensor<nn_type>::zeros({ _amounts });

	set_random_uniform(_kernel, -0.1f, 0.1f);

	weights.append(_kernel);
	weights.append(_bias);
}

void NN_Conv2D::run(const NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output) {
	const NN_Tensor<nn_type>& m_input = input[0].val();
	NN_Tensor<nn_type>& m_output = output[0].val();

	const NN_Shape4D in = m_input.get_shape().get_4dims();
	const NN_Shape4D out = m_output.get_shape().get_4dims();
	const NN_Shape4D k = _kernel.get_shape().get_4dims();

	if (_pad == "same") {
		NN_Shape4D pad = in;

		if (_strides[0] == 1)
			pad._w = in._w - 1 + k._w;
		else
			pad._w = (in._w / _strides[0]) + k._w;

		if (_strides[1] == 1)
			pad._h = in._h - 1 + k._h;
		else
			pad._h = (in._h / _strides[1]) + k._h;

		NN_Tensor<nn_type> pad_tensor;

		if (pad._h != in._h || pad._w != in._w) {
			pad_tensor.resize({ pad._n, pad._h, pad._w, pad._c });

			padding_dilation(
				m_input,
				pad_tensor,
				_strides[1],
				_strides[0],
				_strides[1] == 1 ? (pad._w - in._w) / 2 : 0,
				_strides[0] == 1 ? (pad._h - in._h) / 2 : 0
			);
		}
		else pad_tensor = m_input;


	}
}

NN_Backward* NN_Conv2D::create_backward(std::vector<bool>& mask) {
	return new NN_dConv2D(*this);
}

NN_List<NN_Tensor<nn_type>> NN_Conv2D::get_weight() {
	return { _kernel, _bias };
}



NN_dConv2D::NN_dConv2D(NN_Conv2D& layer) :
	NN_Backward_t(layer)
{

}

void NN_dConv2D::run(
	const NN_List<NN_Tensor<nn_type>>& input,
	const NN_List<NN_Tensor<nn_type>>& doutput,
	NN_List<NN_Tensor<nn_type>>& dinput
) {

}

NN_Optimizer* NN_dConv2D::create_optimizer(const NN_Optimizer& optimizer) {

}