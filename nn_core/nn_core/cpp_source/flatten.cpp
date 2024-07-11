#include "flatten.h"
#include "../cuda_source/cuda_misc.cuh"


/******************************************/
/*                                        */
/*                NN_Flatten              */
/*                                        */
/******************************************/

NN_Flatten::NN_Flatten(const char* name) :
	NN_Layer(name)
{
}

void NN_Flatten::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NC in = input_shape[0].val().get_nc();

	output_shape.append(NN_Shape({ in.n, in.c }));
}

void NN_Flatten::build(const NN_List<NN_Shape>& input_shape, std::vector<GpuTensor<nn_type>>& weights) {
	
}

void NN_Flatten::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0].val();
	GpuTensor<nn_type>& m_output = output[0].val();

	int shape_ranks = m_input.get_shape().get_len();

	if (shape_ranks > 2) {
		std::vector<uint> ranks(shape_ranks, 0);

		for (uint i = 0; i < shape_ranks; ++i) ranks[i] = i;

		uint c_tmp = ranks[shape_ranks - 3];
		ranks[shape_ranks - 3] = ranks[shape_ranks - 2];
		ranks[shape_ranks - 2] = ranks[shape_ranks - 1];
		ranks[shape_ranks - 1] = c_tmp;

		transpose(m_input, m_output, ranks);
	}
	else m_output = m_input;
}

NN_Backward* NN_Flatten::create_backward(NN_Optimizer& optimizer, std::vector<bool>& mask) {
	return new NN_dFlatten(*this, optimizer);
}


/******************************************/
/*                                        */
/*               NN_dFlatten              */
/*                                        */
/******************************************/

NN_dFlatten::NN_dFlatten(NN_Flatten& flatten, NN_Optimizer& optimizer) :
	NN_Backward(optimizer),
	_flatten(flatten)
{

}

void NN_dFlatten::get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape) {

}

void NN_dFlatten::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}