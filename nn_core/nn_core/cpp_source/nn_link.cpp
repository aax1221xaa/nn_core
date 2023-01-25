#include "nn_link.h"


NN_Link* NN_Link::create_child_link() {
	NN_Link* p_child = new NN_Link;

	p_child->cont.op_layer = cont.op_layer;

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

NN NN_Link::operator()(NN m_prev_link) {
	for (const Link_Param<NN_Link>& p_prev_link : m_prev_link) {
		prev_link.push_back(p_prev_link.link);
		p_prev_link.link->next_link.push_back(this);

		cont.input.push_back(&p_prev_link.p_cont->output);
		cont.in_shape.push_back(&p_prev_link.p_cont->out_shape);

		NN_Tensor_t p_dio = new NN_Tensor;

		p_prev_link.p_cont->d_output.push_back(p_dio);
		cont.d_input.push_back(p_dio);
	}

	NN p(this);

	return p;
}

void NN_Link::inner_link(NN_Link* p_prev) {
	prev_link.push_back(p_prev);
	p_prev->next_link.push_back(this);

	cont.input.push_back(&p_prev->cont.output);
	cont.in_shape.push_back(&p_prev->cont.out_shape);

	NN_Tensor_t p_dio = new NN_Tensor;

	cont.d_input.push_back(p_dio);
	p_prev->cont.d_output.push_back(p_dio);
}

NN_Link* NN_Link::get_output_info(NN_Link* p_next) {
	return this;
}

NN_Link* NN_Link::get_input_info(NN_Link* p_prev) {
	return this;
}