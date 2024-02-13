#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"



/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

class NN_Backward {
public:
	virtual ~NN_Backward();
	virtual void set_dio(
		std::vector<nn_shape*>& in_shape,
		std::vector<DeviceTensor<nn_type>*>& d_outputs, 
		std::vector<DeviceTensor<nn_type>*>& d_inputs
	);
	virtual void run_backward(
		cudaStream_t* s,
		std::vector<DeviceTensor<nn_type>*>& inputs,
		DeviceTensor<nn_type>& outputs,
		DeviceTensor<nn_type>& d_output, 
		std::vector<DeviceTensor<nn_type>*>& d_input
	);
};

class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();

	virtual void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) = 0;
	virtual void build(std::vector<nn_shape*>& input_shape);
	virtual void set_io(nn_shape& out_shape, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output);
	virtual void run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) = 0;
	virtual NN_Backward* create_backward(NN_Optimizer& optimizer);
};