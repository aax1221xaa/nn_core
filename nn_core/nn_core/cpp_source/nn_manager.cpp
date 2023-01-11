#include "nn_manager.h"


bool NN_Manager::init_flag = false;
vector<NN_Layer*> NN_Manager::reg_layers;
vector<NN_Link*> NN_Manager::reg_links;

void NN_Manager::add_layer(NN_Layer* layer) {
	reg_layers.push_back(layer);
}

void NN_Manager::add_link(NN_Link* link) {
	reg_links.push_back(link);
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

vector<NN_Link*> NN_Manager::get_links() {
	return reg_links;
}

void NN_Manager::clear_select_flag() {
	for (NN_Link* p : reg_links) p->is_selected = false;
}

NN_Manager::NN_Manager() {
	init_flag = true;
}

NN_Manager::~NN_Manager() {
	clear_layers();
	clear_links();
}