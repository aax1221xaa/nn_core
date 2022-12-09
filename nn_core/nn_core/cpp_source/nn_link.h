#pragma once
#include "nn_base_layer.h"
#include "nn_manager.h"



class NN_Link {
public:
	static vector<NN_Ptr<NN_Link>> total_links;

	static void add_link(NN_Ptr<NN_Link>& link);
	static void clear_select();
	static void clear_links();

public:
	bool is_selected;

	vector<NN_Ptr<NN_Link>> m_prev;
	vector<NN_Ptr<NN_Link>> m_next;

	NN_Ptr<NN_Link> parent;
	vector<NN_Ptr<NN_Link>> child;

	vector<NN_Ptr<NN_Tensor>> input;
	NN_Ptr<NN_Tensor> output;

	NN_Ptr<NN_Tensor> d_input;
	vector<NN_Ptr<NN_Tensor>> d_output;

	vector<Dim*> input_shape;
	Dim output_shape;

	NN_Ptr<NN_Layer> m_layer;

	NN_Link(NN_Ptr<NN_Layer>& layer);
	NN_Link(NN_Ptr<NN_Link>& parent_link);

	NN_Ptr<NN_Link> operator()(NN_Ptr<NN_Link>& prev_link);
	NN_Ptr<NN_Link> operator()(vector<NN_Ptr<NN_Link>>& prev_link);
};