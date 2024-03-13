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

void NN_Dense::test(const std::vector<Tensor<nn_type>>& in_val, std::vector<Tensor<nn_type>>& out_val) {
	/*
	[-1, h, w, c]

	input = [n, h * w * c] ( [n, c_in] )
	weight = [c_in, c_out]
	output = [n, c_out]
	*/
	
	const Tensor<nn_type>& input = in_val[0];
	Tensor<nn_type> output = zeros_like<nn_type>(in_val[0]);

	out_val.push_back(output);

	nn_type* p_input = input.get_data();
	nn_type* p_output = output.get_data();

	for (size_t i = 0; i < output.get_len(); ++i) p_output[i] = p_input[i] + 1.f;
}



/**********************************************/
/*                                            */
/*                  NN_Concat                 */
/*                                            */
/**********************************************/

NN_Concat::NN_Concat(const char* name) :
	NN_Layer(name)
{
}

void NN_Concat::test(const std::vector<Tensor<nn_type>>& in_val, std::vector<Tensor<nn_type>>& out_val) {
	nn_type* input_1 = in_val[0].get_data();
	nn_type* input_2 = in_val[1].get_data();
	
	Tensor<nn_type> output = zeros_like<nn_type>(in_val[0]);
	nn_type* p_output = output.get_data();

	out_val.push_back(output);

	for (size_t i = 0; i < output.get_len(); ++i) p_output[i] = input_1[i] + input_2[i];
}