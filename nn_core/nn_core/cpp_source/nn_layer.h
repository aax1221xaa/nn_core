#pragma once
#include "nn_base_layer.h"


/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

class NN_Dense : public NN_Layer {
public:
	GpuTensor<nn_type> _weight;
	GpuTensor<nn_type> _bias;
	const int _amounts;

	NN_Dense(const int amounts, const char* name);

	nn_shape calculate_output_size(nn_shape& input_shape);
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);
};
