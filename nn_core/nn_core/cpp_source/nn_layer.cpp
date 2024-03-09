#include "nn_layer.h"


/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

NN_Dense::NN_Dense(const int amounts, const char* name) :
	NN_Layer(name),
	_amounts(amounts)
{
}

nn_shape NN_Dense::calculate_output_size(nn_shape& input_shape) {
	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/
	return { {input_shape[0][0], _amounts} };
}

void NN_Dense::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {

}

void NN_Dense::run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) {

}

NN_BackPropLayer* NN_Dense::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}