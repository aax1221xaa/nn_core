#pragma once
#include "nn_base_layer.h"


template <class _T>
struct NN_Coupler {
	_T* link;
	NN_Tensor* output;
	NN_Tensor* d_input;
	Dim* out_size;
};

class NN_Link {
public:
	bool is_selected;
	bool trainable;

	vector<NN_Link*> prev_link;
	vector<NN_Link*> next_link;

	NN_Link* parent;
	vector<NN_Link*> child;

	vector<NN_Tensor*> input;
	vector<NN_Tensor*> d_output;
	vector<Dim*> input_shape;

	NN_Tensor output;
	NN_Tensor d_input;
	Dim output_shape;

	NN_Layer* op_layer;

	NN_Link(NN_Layer* p_layer);
	NN_Link(NN_Link* parent_link);
	~NN_Link();

	virtual NN_Vec<NN_Coupler<NN_Link>> operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link);
	void operator()(NN_Link* m_prev_link);
};