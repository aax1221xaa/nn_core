#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"



/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

class NN_BackPropLayer {
public:
	virtual ~NN_BackPropLayer();
	virtual void set_dio(
		std::vector<nn_shape>& in_shape,
		std::vector<GpuTensor<nn_type>>& d_outputs,
		std::vector<GpuTensor<nn_type>>& d_inputs
	);
	virtual void run_backprop(
		std::vector<cudaStream_t>& s,
		std::vector<GpuTensor<nn_type>>& inputs,
		GpuTensor<nn_type>& outputs,
		GpuTensor<nn_type>& d_output,
		std::vector<GpuTensor<nn_type>>& d_input
	);
};

class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();
	virtual nn_shape calculate_output_size(nn_shape& input_shape) = 0;
	virtual void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	virtual void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) = 0;
	virtual NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);
};