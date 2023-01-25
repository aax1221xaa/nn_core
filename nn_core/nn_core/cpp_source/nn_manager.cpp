#include "nn_manager.h"


bool NN_Manager::init_flag = false;
vector<NN_Layer*> NN_Manager::reg_layers;
vector<NN_Link*> NN_Manager::reg_links;
vector<NN_Tensor_t> NN_Manager::reg_tensor;
vector<NN_Tensor_t> NN_Manager::reg_weight;

void NN_Manager::add_layer(NN_Layer* layer) {
	reg_layers.push_back(layer);
}

void NN_Manager::add_link(NN_Link* link) {
	reg_links.push_back(link);
}

void NN_Manager::add_tensor(NN_Tensor_t tensor) {
	reg_tensor.push_back(tensor);
}

void NN_Manager::add_weight(NN_Tensor_t weight) {
	reg_weight.push_back(weight);
}

void NN_Manager::clear_layers() {
	for (NN_Layer* p : reg_layers) delete p;
	reg_layers.clear();
}

void NN_Manager::clear_links() {
	for (NN_Link* p : reg_links) {
		delete p;
	}
	reg_links.clear();
}

void NN_Manager::clear_tensors() {
	for (NN_Tensor_t p : reg_tensor) delete p;
	reg_tensor.clear();
}

void NN_Manager::clear_weights() {
	for (NN_Tensor_t p : reg_weight) delete p;
	reg_weight.clear();
}

vector<NN_Link*> NN_Manager::get_links() {
	return reg_links;
}

void NN_Manager::clear_select_flag() {
	for (NN_Link* p : reg_links) p->is_selected = false;
}

NN_Manager::NN_Manager() {
	init_flag = true;
	try {
		check_cuda(cudaStreamCreate(&NN_Layer::stream));
	}
	catch (Exception& e) {
		e.Put();
	}
}

NN_Manager::~NN_Manager() {
	try {
		clear_layers();
		clear_links();
		clear_tensors();
		clear_weights();
		check_cuda(cudaStreamDestroy(NN_Layer::stream));
	}
	catch (Exception& e) {
		e.Put();
	}
}