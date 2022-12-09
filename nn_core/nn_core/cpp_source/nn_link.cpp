#include "nn_link.h"


vector<NN_Ptr<NN_Link>> NN_Link::total_links;

void NN_Link::add_link(NN_Ptr<NN_Link>& link) {
	total_links.push_back(link);
}

void NN_Link::clear_select() {
	for (NN_Ptr<NN_Link>& p : total_links) {
		p->is_selected = false;
	}
}

void NN_Link::clear_links() {
	total_links.clear();
}

NN_Link::NN_Link(NN_Ptr<NN_Layer>& layer) :
	m_layer(layer)
{
	is_selected = false;
}

NN_Link::NN_Link(NN_Ptr<NN_Link>& parent_link) :
	parent(parent_link),
	m_layer(parent_link->m_layer)
{
	parent_link->child.push_back(this);
}

NN_Ptr<NN_Link> NN_Link::operator()(NN_Ptr<NN_Link>& prev_link) {
	m_prev.push_back(prev_link);
	prev_link->m_next.push_back(this);

	input.push_back(prev_link->output);
	prev_link->d_output.push_back(d_input);

	input_shape.push_back(&prev_link->output_shape);

	return this;
}

NN_Ptr<NN_Link> NN_Link::operator()(vector<NN_Ptr<NN_Link>>& prev_link) {
	m_prev = prev_link;

	for (NN_Ptr<NN_Link>& p_layer : prev_link) {
		p_layer->m_next.push_back(this);

		input.push_back(p_layer->output);
		p_layer->d_output.push_back(d_input);

		input_shape.push_back(&p_layer->output_shape);
	}

	return this;
}