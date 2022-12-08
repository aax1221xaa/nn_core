#pragma once
#include "nn_base_layer.h"



class NN_Link {
protected:
	static vector<NN_Link*> total_links;

public:
	bool is_selected;

	vector<NN_Link*> m_prev;
	vector<NN_Link*> m_next;

	NN_Link* parent;
	vector<NN_Link*> child;

	vector<NN_Tensor*> input;
	NN_Tensor output;

	NN_Tensor d_input;
	vector<NN_Tensor*> d_output;

	NN_Layer* m_layer;

	NN_Link(NN_Layer* layer);
	NN_Link(NN_Link* parent_link);

	NN_Link* operator()(NN_Link* prev_link);
	NN_Link* operator()(vector<NN_Link*> prev_link);

	static NN_Link* create_link(NN_Link* parent_link = NULL, NN_Layer* layer = NULL);
	static void destroy_links();
};