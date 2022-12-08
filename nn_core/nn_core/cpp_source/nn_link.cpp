#include "nn_link.h"


vector<NN_Link*> NN_Link::total_links;

NN_Link::NN_Link(NN_Layer* layer) {
	is_selected = false;
	m_layer = layer;
	parent = NULL;
}

NN_Link::NN_Link(NN_Link* parent_link) {
	parent = parent_link;
	parent_link->child.push_back(this);

	m_layer = parent_link->m_layer;
}

NN_Link* NN_Link::operator()(NN_Link* prev_link) {
	m_prev.push_back(prev_link);
	prev_link->m_next.push_back(this);

	input.push_back(&prev_link->output);
	prev_link->d_output.push_back(&d_input);

	return this;
}

NN_Link* NN_Link::operator()(vector<NN_Link*> prev_link) {
	m_prev = prev_link;

	for (NN_Link* p_layer : prev_link) {
		p_layer->m_next.push_back(this);

		input.push_back(&p_layer->output);
		p_layer->d_output.push_back(&d_input);
	}

	return this;
}

NN_Link* NN_Link::create_link(NN_Link* parent_link, NN_Layer* layer) {
	NN_Link* p_child_link = NULL;

	if (parent_link) {
		p_child_link = new NN_Link(parent_link);
	}
	else {
		p_child_link = new NN_Link(layer);
	}

	return p_child_link;
}

void NN_Link::destroy_links() {
	for (NN_Link* p_link : total_links) {
		nn_tensor_free(p_link->output);
		nn_tensor_free(p_link->d_input);

		delete p_link;
	}

	total_links.clear();
}