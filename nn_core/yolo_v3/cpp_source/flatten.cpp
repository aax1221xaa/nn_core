#include "flatten.h"
#include "../cuda_source/cuda_misc.cuh"


/******************************************/
/*                                        */
/*                NN_Flatten              */
/*                                        */
/******************************************/

NN_Flatten::NN_Flatten(const std::string& name) :
	NN_Layer(name, "flatten")
{
}

void NN_Flatten::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& in = input_shape[0].val();
	int n_nodes = 1;

	for (NN_Shape::c_iterator iter = in.begin() + 1; iter != in.end(); ++iter) n_nodes *= *iter;

	output_shape.append(NN_Shape({ in[0], n_nodes }));
}

void NN_Flatten::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	
}

void NN_Flatten::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {

}

NN_Backward* NN_Flatten::create_backward(std::vector<bool>& mask) {
	return new NN_dFlatten(*this);
}

void NN_Flatten::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& in_tensor = input[0].val();
	const NN_Shape& out_shape = output_shape[0].val();
	
	GpuTensor<nn_type> out_tensor(in_tensor, out_shape);

	output.append(out_tensor);
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
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}