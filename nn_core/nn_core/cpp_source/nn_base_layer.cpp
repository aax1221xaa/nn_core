#include "nn_base_layer.h"


vector<NN_Layer*> NN_Layer::total_layers;

NN_Layer::NN_Layer() {
	trainable = false;
}

void NN_Layer::add_layer(NN_Layer* layer) {
	total_layers.push_back(layer);
}

void NN_Layer::destroy_layers() {
	for (NN_Layer* p_layer : total_layers) {
		delete p_layer;
	}

	total_layers.clear();
}