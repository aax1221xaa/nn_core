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


/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/

NN_Test::NN_Test(const char* name) :
	NN_Layer(name)
{
}

void NN_Test::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	output_shape.clear();

	NN_Shape out_shape;
	input_shape[0].copy_to(out_shape);
	
	output_shape.push_back(out_shape);
}

void NN_Test::build(const std::vector<NN_Shape>& input_shape) {
	std::cout << "build: " << _layer_name << std::endl;
}

void NN_Test::run_forward(const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	gpu_to_gpu(input[0], output[0]);
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

void NN_Concat::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	output_shape.clear();

	for (int i = 0; i < input_shape[0].get_size(); ++i) {
		int n = input_shape[0][i];
		for (const NN_Shape& shape : input_shape) {
			if (shape[i] != n) {
				ErrorExcept(
					"[NN_Concat::get_output_shape] input shapes are differents."
				);
			}
		}
	}

	NN_Shape out_shape;
	input_shape[0].copy_to(out_shape);

	output_shape.push_back(out_shape);
}

void NN_Concat::build(const std::vector<NN_Shape>& input_shape) {
	std::cout << "build: " << _layer_name << std::endl;
}

void NN_Concat::run_forward(const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	Tensor<nn_type> m_input(input[0].get_shape());
	Tensor<nn_type> m_output = zeros_like<nn_type>(m_input);

	for (const GpuTensor<nn_type>& p_input : input) {
		gpu_to_host(p_input, m_input);
		
		const size_t size = calculate_length(m_input.get_shape());
		const nn_type* input_data = m_input.get_data();
		nn_type* output_data = m_output.get_data();

		for (size_t i = 0; i < size; ++i) output_data[i] += input_data[i];
	}

	host_to_gpu(m_output, output[0]);
}