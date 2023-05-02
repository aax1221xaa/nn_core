#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();

	virtual void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) = 0;
	virtual void build(std::vector<nn_shape*>& input_shape);
	virtual void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) = 0;
	virtual void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) = 0;
};