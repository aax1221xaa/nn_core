#include "nn_ops.h"
#include "../cuda_source/cuda_misc.cuh"


/**********************************************/
/*                                            */
/*                    NN_Ops                  */
/*                                            */
/**********************************************/

NN_Ops::NN_Ops(const std::string& layer_name) :
	NN_Layer(layer_name, "ops"),
	_scalar(0.f)
{

}

NN_Ops::NN_Ops(nn_type scalar, const std::string& layer_name) :
	NN_Layer(layer_name, "ops"),
	_scalar(scalar)
{

}

NN_Ops::NN_Ops(const NN_Ops& p) :
	NN_Layer(p),
	_scalar(p._scalar)
{

}

NN_Ops::NN_Ops(NN_Ops&& p) :
	NN_Layer(p),
	_scalar(p._scalar)
{

}

const NN_Ops& NN_Ops::operator=(const NN_Ops& p) {
	if (this == &p) return *this;

	_scalar = p._scalar;

	return *this;
}

const NN_Ops& NN_Ops::operator=(NN_Ops&& p) {
	_scalar = p._scalar;

	return *this;
}

void NN_Ops::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_Ops::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const NN_Shape shape = input[0].val().get_shape();
	GpuTensor<nn_type> out_tensor(shape);

	output.append(out_tensor);
}


/**********************************************/
/*                                            */
/*                    NN_Add                  */
/*                                            */
/**********************************************/

NN_Add::NN_Add(const std::string& layer_name) :
	NN_Ops(0.f, layer_name)
{

}

NN_Add::NN_Add(nn_type scalar, const std::string& layer_name) :
	NN_Ops(scalar, layer_name)
{

}

NN_Add::NN_Add(const NN_Add& p) :
	NN_Ops(p)
{

}

NN_Add::NN_Add(NN_Add&& p) :
	NN_Ops(p)
{

}

void NN_Add::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& c_output = output[0].val();

	if (input.size() > 1) {
		const GpuTensor<nn_type>& a_input = input[0].val();
		const GpuTensor<nn_type>& b_input = input[1].val();

		add_tensor(a_input, b_input, c_output);
	}
	else {
		const GpuTensor<nn_type>& a_input = input[0].val();

		add_tensor(a_input, _scalar, c_output);
	}
}


/**********************************************/
/*                                            */
/*                    NN_Sub                  */
/*                                            */
/**********************************************/

NN_Sub::NN_Sub(const std::string& layer_name) :
	NN_Ops(0.f, layer_name)
{

}

NN_Sub::NN_Sub(nn_type scalar, const std::string& layer_name) :
	NN_Ops(scalar, layer_name)
{

}

NN_Sub::NN_Sub(const NN_Sub& p) :
	NN_Ops(p)
{

}

NN_Sub::NN_Sub(NN_Sub&& p) :
	NN_Ops(p)
{

}

void NN_Sub::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& c_output = output[0].val();

	if (input.size() > 1) {
		const GpuTensor<nn_type>& a_input = input[0].val();
		const GpuTensor<nn_type>& b_input = input[1].val();

		sub_tensor(a_input, b_input, c_output);
	}
	else {
		const GpuTensor<nn_type>& a_input = input[0].val();

		sub_tensor(a_input, _scalar, c_output);
	}
}


/**********************************************/
/*                                            */
/*                    NN_Mul                  */
/*                                            */
/**********************************************/

NN_Mul::NN_Mul(const std::string& layer_name) :
	NN_Ops(0.f, layer_name)
{

}

NN_Mul::NN_Mul(nn_type scalar, const std::string& layer_name) :
	NN_Ops(scalar, layer_name)
{

}

NN_Mul::NN_Mul(const NN_Mul& p) :
	NN_Ops(p)
{

}

NN_Mul::NN_Mul(NN_Mul&& p) :
	NN_Ops(p)
{

}

void NN_Mul::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& c_output = output[0].val();

	if (input.size() > 1) {
		const GpuTensor<nn_type>& a_input = input[0].val();
		const GpuTensor<nn_type>& b_input = input[1].val();

		mul_tensor(a_input, b_input, c_output);
	}
	else {
		const GpuTensor<nn_type>& a_input = input[0].val();

		mul_tensor(a_input, _scalar, c_output);
	}
}


/**********************************************/
/*                                            */
/*                    NN_Div                  */
/*                                            */
/**********************************************/

NN_Div::NN_Div(const std::string& layer_name) :
	NN_Ops(0.f, layer_name)
{

}

NN_Div::NN_Div(nn_type scalar, const std::string& layer_name) :
	NN_Ops(scalar, layer_name)
{

}

NN_Div::NN_Div(const NN_Div& p) :
	NN_Ops(p)
{

}

NN_Div::NN_Div(NN_Div&& p) :
	NN_Ops(p)
{

}

void NN_Div::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& c_output = output[0].val();

	if (input.size() > 1) {
		const GpuTensor<nn_type>& a_input = input[0].val();
		const GpuTensor<nn_type>& b_input = input[1].val();

		div_tensor(a_input, b_input, c_output);
	}
	else {
		const GpuTensor<nn_type>& a_input = input[0].val();

		div_tensor(a_input, _scalar, c_output);
	}
}