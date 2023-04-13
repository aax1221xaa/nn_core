#include "nn_link.h"


#if !(FIX_MODE)

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
	NN_Link* m_prev_node = prev_node[0]._link;
	int n_prev_node = prev_node[0]._output_index;

	m_prev_node->set_next_node(this, n_prev_node);
	_prev.push_back(m_prev_node);
	_input.push_back(&m_prev_node->get_output(n_prev_node));
	m_prev_node->get_d_output(n_prev_node).push_back(&_d_input);

	return { Layer_Ptr<NN_Link> { this, 0 } };
}

Layer_t NN_Link::operator()(initializer_list<Layer_t> prev_node) {
	for (Layer_t p_prev_node : prev_node) {
		NN_Link* m_prev_node = p_prev_node[0]._link;
		int n_prev_node = p_prev_node[0]._output_index;

		m_prev_node->set_next_node(this, n_prev_node);
		_prev.push_back(m_prev_node);
		_input.push_back(&m_prev_node->get_output(n_prev_node));
		m_prev_node->get_d_output(n_prev_node).push_back(&_d_input);
	}

	return { Layer_Ptr<NN_Link> { this, 0 } };
}

int NN_Link::get_node_index(NN_Link* next_node) {
	return 0;
}

void NN_Link::set_next_node(NN_Link* next_node, int node_index) {
	_next.push_back(next_node);
}

NN_Tensor& NN_Link::get_output(int node_index) {
	return _output;
}

vector<NN_Tensor*>& NN_Link::get_d_output(int node_index) {
	return _d_output;
}

void NN_Link::link_prev_child() {
	for (NN_Link* p_prev : _parent->_prev) {
		NN_Link* p_prev_child = NN_Link::get_child(p_prev);

		if (p_prev_child) {
			int n_prev_child = p_prev->get_node_index(this->_parent);

			p_prev_child->set_next_node(this, n_prev_child);
			_prev.push_back(p_prev_child);
			_input.push_back(&p_prev_child->get_output(n_prev_child));
			p_prev_child->get_d_output(n_prev_child).push_back(&_d_input);
		}
	}
}

NN_Link* NN_Link::get_child(NN_Link* current_parent) {
	NN_Link* select_child = NULL;

	for (NN_Link* p_child : current_parent->_child) {
		if (p_child->is_selected) {
			select_child = p_child;
			break;
		}
	}

	return select_child;
}

#else

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
	NN_Link* m_prev_node = prev_node[0].get()._link;
	int n_prev_node = prev_node[0].get()._output_index;

	m_prev_node->set_next_node(this, n_prev_node);
	_prev.push_back(m_prev_node);
	_input.push_back(&m_prev_node->get_output(n_prev_node));
	m_prev_node->get_d_output(n_prev_node).push_back(&_d_input);

	return { Layer_Ptr<NN_Link> { this, 0 } };
}

Layer_t NN_Link::operator()(std::initializer_list<Layer_t> prev_node) {
	for (Layer_t p_prev_node : prev_node) {
		NN_Link* m_prev_node = p_prev_node[0].get()._link;
		int n_prev_node = p_prev_node[0].get()._output_index;

		m_prev_node->set_next_node(this, n_prev_node);
		_prev.push_back(m_prev_node);
		_input.push_back(&m_prev_node->get_output(n_prev_node));
		m_prev_node->get_d_output(n_prev_node).push_back(&_d_input);
	}

	return { Layer_Ptr<NN_Link> { this, 0 } };
}

int NN_Link::get_node_index(NN_Link* next_node) {
	return 0;
}

void NN_Link::set_next_node(NN_Link* next_node, int node_index) {
	_next.push_back(next_node);
}

NN_Tensor& NN_Link::get_output(int node_index) {
	return _output;
}

std::vector<NN_Tensor*>& NN_Link::get_d_output(int node_index) {
	return _d_output;
}

void NN_Link::link_prev_child() {
	for (NN_Link* p_prev : _parent->_prev) {
		NN_Link* p_prev_child = NN_Link::get_child(p_prev);

		if (p_prev_child) {
			int n_prev_child = p_prev->get_node_index(this->_parent);

			p_prev_child->set_next_node(this, n_prev_child);
			_prev.push_back(p_prev_child);
			_input.push_back(&p_prev_child->get_output(n_prev_child));
			p_prev_child->get_d_output(n_prev_child).push_back(&_d_input);
		}
	}
}

NN_Link* NN_Link::get_child(NN_Link* current_parent) {
	NN_Link* select_child = NULL;

	for (NN_Link* p_child : current_parent->_child) {
		if (p_child->is_selected) {
			select_child = p_child;
			break;
		}
	}

	return select_child;
}

#endif