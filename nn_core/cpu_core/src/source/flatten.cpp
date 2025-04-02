#include "../header/flatten.h"



/******************************************/
/*                                        */
/*                NN_Flatten              */
/*                                        */
/******************************************/

NN_Flatten::NN_Flatten(const std::string& name) :
	NN_Layer(name)
{
}

void NN_Flatten::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& in = input_shape[0].val();
	int n_nodes = 1;

	for (NN_Shape::c_iter iter = in.begin() + 1; iter != in.end(); ++iter) n_nodes *= *iter;

	output_shape.append(NN_Shape({ in[0], n_nodes }));
}

void NN_Flatten::run(const NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output) {
	const NN_Tensor<nn_type>& m_input = input[0].val();
	NN_Tensor<nn_type>& m_output = output[0].val();

	NN_Shape shape = m_input.get_shape();
	NN_Shape reshape({ 1, 1 });

	reshape[0] = shape[0];

	for (NN_Shape::iter i = shape.begin() + 1; i < shape.end(); ++i) {
		reshape[1] *= *i;
	}

	NN_Tensor<nn_type> tmp(shape);

	tmp = m_input;
	m_output = NN_Tensor<nn_type>(tmp.get_shared_ptr(), reshape);
}

NN_Backward* NN_Flatten::create_backward(std::vector<bool>& mask) {
	return new NN_dFlatten(*this);
}


/******************************************/
/*                                        */
/*               NN_dFlatten              */
/*                                        */
/******************************************/

NN_dFlatten::NN_dFlatten(NN_Flatten& flatten) :
	NN_Backward_t(flatten)
{

}

void NN_dFlatten::run(
	const NN_List<NN_Tensor<nn_type>>& input,
	const NN_List<NN_Tensor<nn_type>>& doutput,
	NN_List<NN_Tensor<nn_type>>& dinput
) {

}