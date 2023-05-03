#include "nn_base_layer.h"


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

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