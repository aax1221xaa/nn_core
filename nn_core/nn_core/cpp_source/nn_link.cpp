#include "nn_link.h"


NN_Link::NN_Link(NN_Ptr<NN_Layer> p_layer) :
	layer(p_layer),
	output(new NN_Tensor),
	d_input(new NN_Tensor),
	parent(NULL)
{
	is_selected = false;
	trainable = true;
}

NN_Link::NN_Link(NN_Ptr<NN_Link> parent_link) :
	parent(parent_link),
	layer(parent_link->layer),
	output(new NN_Tensor),
	d_input(new NN_Tensor)
{
	is_selected = true;
	trainable = true;

	parent_link->child.push_back(this);
}

NN_Coupler<NN_Link> NN_Link::operator()(NN_Coupler<NN_Link>& m_prev_link) {
	prev_link = m_prev_link.object;

	for (NN_Ptr<NN_Link>& p_prev : m_prev_link.object) {
		p_prev->next_link.push_back(this);
	}

	return NN_Ptr<NN_Link>(this);
}

void NN_Link::operator()(NN_Ptr<NN_Link>& m_prev_link) {
	prev_link.push_back(m_prev_link);
	m_prev_link->next_link.push_back(this);
}

NN_Ptr<NN_Tensor> NN_Link::get_prev_output(NN_Ptr<NN_Link>& p_current) {
	return this->output;
}

NN_Ptr<NN_Tensor> NN_Link::get_next_dinput(NN_Ptr<NN_Link>& p_current) {
	return this->d_input;
}

Dim NN_Link::get_next_output_shape(NN_Ptr<NN_Link>& p_current){
	return this->output_shape;
}