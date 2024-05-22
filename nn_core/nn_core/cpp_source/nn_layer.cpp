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
	const NC nc = input[0].get_shape().get_nc();

	output[0] = input[0];
	output[0].reshape({ nc.n, nc.c });
}


