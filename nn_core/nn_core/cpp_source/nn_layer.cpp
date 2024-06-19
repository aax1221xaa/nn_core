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
	Tensor<nn_type> h_input(input[0].get_shape());
	const NC nc = h_input.get_shape().get_nc();

	h_input = input[0];

	Tensor<nn_type> t_input = h_input.transpose({ 0, 2, 3, 1 });
	Tensor<nn_type> tmp(t_input.get_shape());

	tmp = t_input;

	const nn_type* p_src = tmp.get_ptr();
	nn_type* p_dst = output[0].get_ptr();
	const size_t len = tmp.get_shape().total_size();

	check_cuda(cudaMemcpy(p_dst, p_src, sizeof(nn_type) * len, cudaMemcpyHostToDevice));

	//Tensor<nn_type> tmp = t_input[0];

	//std::cout << std::endl << tmp;
}


