#include "nn_operator.h"


/**********************************************/
/*                                            */
/*					 NN_Div                   */
/*                                            */
/**********************************************/

NN_Div::NN_Div() :
	NN_Operator("div"),
	_val(0.f)
{

}

void NN_Div::op(NN_Stream& st, const GpuTensor<nn_type>& a, const GpuTensor<nn_type>& b, GpuTensor<nn_type>& c) {
	Tensor<nn_type> _a(a.get_shape());
	Tensor<nn_type> _b(b.get_shape());
	Tensor<nn_type> _c(c.get_shape());

	_a = a;
	_b = b;

	_c = _a / _b;

	c = _c;
}

void NN_Div::op(NN_Stream& st, const GpuTensor<nn_type>& a, nn_type b, GpuTensor<nn_type>& c) {
	Tensor<nn_type> _a(a.get_shape());
	Tensor<nn_type> _c(c.get_shape());

	_a = a;

	_c = _a / b;

	c = _c;
}

void NN_Div::op(NN_Stream& st, nn_type a, const GpuTensor<nn_type>& b, GpuTensor<nn_type>& c) {
	Tensor<nn_type> _a(b.get_shape());
	Tensor<nn_type> _b(b.get_shape());
	Tensor<nn_type> _c(c.get_shape());

	_a = a;
	_b = b;

	_c = _a / _b;

	c = _c;
}

void NN_Div::set_const_value(nn_type val, int status) {
	_val = val;
	_status = status;
}

nn_type NN_Div::get_val() {
	return _val;
}