#include "nn_manager.h"


cudaStream_t NN_Manager::_stream = NULL;
std::vector<NN_Link*> NN_Manager::_nodes;
std::vector<NN_Layer*> NN_Manager::_layers;

void NN_Manager::add_node(NN_Link* p_node) {
	_nodes.push_back(p_node);
}

void NN_Manager::add_layer(NN_Layer* p_layer) {
	_layers.push_back(p_layer);
}

NN_Manager::NN_Manager() {
	try {
		_stream = NULL;
		check_cuda(cudaStreamCreate(&_stream));
	}
	catch (Exception& e) {
		e.Put();
	}
}

NN_Manager::~NN_Manager() {
	try {
		for (NN_Link* p_node : _nodes) {
			/*
			vector<NN_Link*> del_list;

			for (NN_Link* p_child : p_node->_child) {
				del_list.push_back(p_child);

				while (!del_list.empty()) {
					NN_Link* p_current_child = del_list.front();

					for (NN_Link* p_grand_child : p_current_child->_child) del_list.push_back(p_grand_child);

					delete p_current_child;
					del_list.erase(del_list.begin());
				}
			}
			*/
			delete p_node;
		}

		for (NN_Layer* p_layer : _layers) delete p_layer;
		
		check_cuda(cudaStreamDestroy(_stream));
	}
	catch (Exception& e) {
		e.Put();
	}
}

void NN_Manager::clear_select_mask() {
	for (NN_Link* p : _nodes) p->is_selected = false;
}