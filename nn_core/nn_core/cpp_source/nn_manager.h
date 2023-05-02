#pragma once
#include "nn_link.h"


/**********************************************/
/*                                            */
/*                  NN_Manager                */
/*                                            */
/**********************************************/

class NN_Manager {
public:
	typedef float nn_type;

	static cudaStream_t _stream[STREAMS];
	static std::vector<NN_Link*> _nodes;
	static std::vector<NN_Layer*> _layers;

	static bool condition;

public:
	static void add_node(NN_Link* p_node);
	static void add_layer(NN_Layer* p_layer);

	NN_Manager();
	~NN_Manager();

	static void clear_select_mask();
};
