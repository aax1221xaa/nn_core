#pragma once
#include "nn_link.h"



class NN_Manager {
public:
	static bool init_flag;

	static vector<NN_Layer*> reg_layers;
	static vector<NN_Link*> reg_links;
	static vector<NN_Tensor_t> reg_tensor;
	static vector<NN_Tensor_t> reg_weight;

public:
	static void add_layer(NN_Layer* layer);
	static void add_link(NN_Link* link);
	static void add_tensor(NN_Tensor_t tensor);
	static void add_weight(NN_Tensor_t weight);
	static void clear_layers();
	static void clear_links();
	static void clear_tensors();
	static void clear_weights();
	static vector<NN_Link*> get_links();
	static void clear_select_flag();

	NN_Manager();
	~NN_Manager();
};
