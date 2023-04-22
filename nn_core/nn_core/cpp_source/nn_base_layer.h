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

	virtual nn_shape calculate_output_size(std::vector<nn_shape*>& input_shape) = 0;
	virtual void build(std::vector<nn_shape*>& input_shape);
	virtual NN_Tensor<nn_type> run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input) = 0;
	virtual NN_Tensor<nn_type> run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output) = 0;
};