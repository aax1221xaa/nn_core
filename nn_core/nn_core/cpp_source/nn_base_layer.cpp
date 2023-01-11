#include "nn_base_layer.h"


NN_Layer::NN_Layer(const string& _layer_name) :
	layer_name(_layer_name)
{
}

NN_Layer::~NN_Layer() {

}

void NN_Layer::build(vector<NN_Shape*>& input_shape) {

}