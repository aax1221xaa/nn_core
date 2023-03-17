#include "nn_link.h"


NN_Link* NN_Link::create_child_link() {
	NN_Link* p_child = new NN_Link;

	p_child->op_layer = op_layer;

	p_child->parent = this;
	child.push_back(p_child);

	return p_child;
}

NN_Link::NN_Link() {
	parent = NULL;
	is_selected = false;
	trainable = true;
}

NN_Link::~NN_Link() {

}

NN_Coupler<NN_Link> NN_Link::operator()(NN_Coupler<NN_Link> m_prev_link) {
	for (const Link_Param<NN_Link>& p_prev_link : m_prev_link) {
		prev_link.push_back(p_prev_link.link);
		p_prev_link.link->next_link.push_back(this);

		input.push_back(&p_prev_link.sub_link->output);

		NN_Tensor_t p_dio = new NN_Tensor;
		p_prev_link.sub_link->d_output.push_back(p_dio);
		d_input.push_back(p_dio);

		in_shape.push_back(&p_prev_link.sub_link->out_shape);
	}

	NN_Coupler<NN_Link> p(this);

	return p;
}

void NN_Link::inner_link(NN_Link* m_prev_link) {
	prev_link.push_back(m_prev_link);
	m_prev_link->next_link.push_back(this);

	input.push_back(&m_prev_link->output);
	
	NN_Tensor_t p_dio = new NN_Tensor;
	m_prev_link->d_output.push_back(p_dio);
	d_input.push_back(p_dio);

	in_shape.push_back(&m_prev_link->out_shape);
}

NN_Link* NN_Link::get_prev_ptr(NN_Link* p_current) {
	return this;
}

NN_Link* NN_Link::get_current_ptr(NN_Link* p_prev) {
	return this;
}