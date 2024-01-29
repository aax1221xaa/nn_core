#include "nn_base_layer.h"


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

#ifdef FIX_MODE

NN_Backward::~NN_Backward() {

}

void NN_Backward::set_dio(
	std::vector<nn_shape*>& in_shape,
	std::vector<DeviceTensor<nn_type>*>& d_outputs,
	std::vector<DeviceTensor<nn_type>*>& d_inputs
) {
	for (DeviceTensor<nn_type>* p_dinput : d_inputs) *p_dinput = DeviceTensor<nn_type>::zeros(*in_shape[0]);
}

void NN_Backward::run_backward(
	cudaStream_t* s,
	std::vector<DeviceTensor<nn_type>*>& inputs,
	DeviceTensor<nn_type>& outputs,
	DeviceTensor<nn_type>& d_output,
	std::vector<DeviceTensor<nn_type>*>& d_input
) {

}

NN_Layer::NN_Layer(const char* layer_name) :
	_layer_name(layer_name)
{
}

NN_Layer::~NN_Layer() {

}

void NN_Layer::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Layer::set_io(nn_shape& out_shape, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output) {
	output.set(out_shape);
}

NN_Backward* NN_Layer::create_backward(NN_Optimizer& optimizer) {
	return NULL;
}

#endif


#ifndef FIX_MODE

NN_Layer::NN_Layer(const char* layer_name) :
	_layer_name(layer_name)
{
}

NN_Layer::~NN_Layer() {

}

void NN_Layer::build(std::vector<nn_shape*>& input_shape) {

}

void NN_Layer::set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	output.set(out_shape);
}

#endif