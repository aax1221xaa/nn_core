#include "nn_link.h"


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

NN_Link::NN_Link() :
	is_selected(false),
	trainable(true),
	_forward(NULL),
	_backward(NULL),
	_parent(NULL)
{
}

NN_Link::~NN_Link() {

}

NN_Link* NN_Link::create_child() {
	NN_Link* child_node = new NN_Link;

	child_node->_parent = this;
	child_node->is_selected = true;
	child_node->_forward = _forward;

	_child.push_back(child_node);

	return child_node;
}

Layer_t NN_Link::operator()(Layer_t& prev_node) {
	prev_node[0]._link->_next.push_back(this);
	_prev.push_back(prev_node[0]._link);
	_input.push_back(prev_node[0]._output);
	prev_node[0]._d_output->push_back(&_d_input);

	return { Layer_Ptr<NN_Link> { this, &_output, &_d_output } };
}

Layer_t NN_Link::operator()(initializer_list<Layer_t> prev_node) {
	for (Layer_t p_prev_node : prev_node) {
		p_prev_node[0]._link->_next.push_back(this);
		_prev.push_back(p_prev_node[0]._link);
		_input.push_back(p_prev_node[0]._output);
		p_prev_node[0]._d_output->push_back(&_d_input);
	}

	return { Layer_Ptr<NN_Link> { this, &_output, &_d_output } };
}

void NN_Link::operator()(NN_Link* prev_node) {
	prev_node->_next.push_back(this);
	_prev.push_back(prev_node);
	_input.push_back(&prev_node->_output);
	prev_node->_d_output.push_back(&_d_input);
}

void NN_Link::set_child_link(NN_Link* current_node) {
	NN_Link* current_child = NULL;

	for (NN_Link* p_child : current_node->_child) {
		if (p_child->is_selected) {
			current_child = p_child;
			break;
		}
	}
	for (NN_Link* p_prev : current_node->_prev) {
		for (NN_Link* p_child : p_prev->_child) {
			if (p_child->is_selected) {
				(*current_child)(p_child);
				break;
			}
		}
	}
}