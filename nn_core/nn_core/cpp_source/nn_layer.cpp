#include "nn_layer.h"










/**********************************************/
/*                                            */
/*                   NN_Flat                  */
/*                                            */
/**********************************************/

NN_Flat::NN_Flat(const char* name) :
	NN_Layer(name)
{
}

void NN_Flat::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	const NC in = input_shape[0].get_nc();

	output_shape.push_back({ in.n, in.c });
}

void NN_Flat::build(const std::vector<NN_Shape>& input_shape) {

}

void NN_Flat::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0];
	GpuTensor<nn_type>& m_output = output[0];
	GpuTensor<nn_type> tmp = m_output;

	int shape_ranks = m_input.get_shape().get_len();

	if (shape_ranks > 2) {
		int c_tmp = 0;

		NN_Shape& tmp_shape = tmp.get_shape();

		tmp_shape = m_input.get_shape();
		c_tmp = tmp_shape[shape_ranks - 3];
		tmp_shape[shape_ranks - 3] = tmp_shape[shape_ranks - 2];
		tmp_shape[shape_ranks - 2] = tmp_shape[shape_ranks - 1];
		tmp_shape[shape_ranks - 1] = c_tmp;

		std::cout << std::endl << tmp_shape;
		std::cout << m_output.get_shape();

		transpose(m_input, tmp);
	}
}


