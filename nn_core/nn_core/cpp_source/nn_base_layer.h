#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

typedef vector<vector<int>> shape_type;

class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();

	virtual shape_type calculate_output_size(shape_type& input_shape) = 0;
	virtual void build(shape_type& input_shape);
	virtual NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input) = 0;
	virtual NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output) = 0;
};