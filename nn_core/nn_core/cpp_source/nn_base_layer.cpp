#include "nn_base_layer.h"


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

NN_BackPropLayer::~NN_BackPropLayer() {

}

void NN_BackPropLayer::set_dio(
	std::vector<nn_shape>& in_shape,
	std::vector<GpuTensor<nn_type>>& d_outputs,
	std::vector<GpuTensor<nn_type>>& d_inputs
) {

}

void NN_BackPropLayer::run_backprop(
	std::vector<cudaStream_t>& s,
	std::vector<GpuTensor<nn_type>>& inputs,
	GpuTensor<nn_type>& outputs,
	GpuTensor<nn_type>& d_output,
	std::vector<GpuTensor<nn_type>>& d_input
) {

}

NN_Layer::NN_Layer(const char* layer_name) :
	_layer_name(layer_name)
{
}

NN_Layer::~NN_Layer() {

}

void NN_Layer::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {
	output.set(out_shape);
}

NN_BackPropLayer* NN_Layer::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}