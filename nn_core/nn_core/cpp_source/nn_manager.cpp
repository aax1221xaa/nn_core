#include "nn_manager.h"


cudaStream_t NN_Manager::_stream[STREAMS];
std::vector<NN_Link*> NN_Manager::_nodes;
std::vector<NN_Layer*> NN_Manager::_layers;
bool NN_Manager::condition = false;

void NN_Manager::add_node(NN_Link* p_node) {
	_nodes.push_back(p_node);
}

void NN_Manager::add_layer(NN_Layer* p_layer) {
	_layers.push_back(p_layer);
}

NN_Manager::NN_Manager() {
	try {
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamCreate(&_stream[i]));
		
		condition = true;
	}
	catch (const Exception& e) {
		condition = false;
		
		for (int i = 0; i < STREAMS; ++i) {
			cudaStreamDestroy(_stream[i]);
			_stream[i] = NULL;
		}
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
		for (int i = 0; i < STREAMS; ++i) {
			check_cuda(cudaStreamDestroy(_stream[i]));
			_stream[i] = NULL;
		}
	}
	catch (Exception& e) {
		e.Put();
	}

	condition = false;
}

void NN_Manager::clear_mark() {
	for (NN_Link* p : _nodes) {
		p->_mark = 0;
		p->_p_link = NULL;
	}
}