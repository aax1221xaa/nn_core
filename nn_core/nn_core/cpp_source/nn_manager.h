#pragma once
#include "nn_link.h"



class NN_Manager {
public:
	static cudaStream_t _stream;
	static vector<NN_Link*> _nodes;
	static vector<NN_Layer*> _layers;

public:
	static void add_node(NN_Link* p_node);
	static void add_layer(NN_Layer* p_layer);

	NN_Manager();
	~NN_Manager();

	static void clear_select_mask();
};
