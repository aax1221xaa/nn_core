#include "nn_link.h"


NN_Link::NN_Link(NN_Layer* p_layer) :
	op_layer(p_layer),
	parent(NULL)
{
	is_selected = false;
	trainable = true;
}

NN_Link::NN_Link(NN_Link* parent_link) :
	parent(parent_link),
	op_layer(parent_link->op_layer)
{
	is_selected = false;
	trainable = true;
}

NN_Link::~NN_Link() {
	delete op_layer;
}

NN_Vec<NN_Coupler<NN_Link>> NN_Link::operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link) {
	for (const NN_Coupler<NN_Link>& p_coupler : m_prev_link.arr) {
		prev_link.push_back(p_coupler.link);
		input.push_back(p_coupler.output);
		d_output.push_back(p_coupler.d_input);
		input_shape.push_back(p_coupler.out_size);
		p_coupler.link->next_link.push_back(this);
	}

	NN_Coupler<NN_Link> p;

	p.link = this;
	p.output = &output;
	p.d_input = &d_input;
	p.out_size = &output_shape;

	return p;
}