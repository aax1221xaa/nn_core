#include "nn_link.h"


NN_Link* NN_Link::create_child_link() {
	NN_Link* p_child = new NN_Link;

	p_child->op_layer = this->op_layer;
	p_child->parent = this;
	child.push_back(p_child);

	return p_child;
}

NN_Link::NN_Link() {
	op_layer = NULL;
	parent = NULL;
	is_selected = false;
	trainable = true;
}

NN_Link::~NN_Link() {
	
}

NN_Coupler<NN_Link> NN_Link::operator()(NN_Coupler<NN_Link> m_prev_link) {
	for (const Link_Param<NN_Link>& p_coupler : m_prev_link) {
		prev_link.push_back(p_coupler.link);
		input.push_back(p_coupler.output);
		d_output.push_back(p_coupler.d_input);
		in_shape.push_back(p_coupler.out_size);
		p_coupler.link->next_link.push_back(this);
	}

	NN_Coupler<NN_Link> p(this);

	return p;
}

void NN_Link::operator()(NN_Link* m_prev_link) {
	prev_link.push_back(m_prev_link);
	m_prev_link->next_link.push_back(this);
	input.push_back(&m_prev_link->output);
	d_output.push_back(&m_prev_link->d_input);
	in_shape.push_back(&m_prev_link->out_shape);
}