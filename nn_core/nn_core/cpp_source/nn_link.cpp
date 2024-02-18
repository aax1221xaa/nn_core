#include "nn_link.h"


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

NN_Link::NN_Link() :
	_mark(0),
	_p_link(NULL),
	trainable(true),
	_forward(NULL),
	_backward(NULL)
{
}

NN_Link::~NN_Link() {

}

NN_Link* NN_Link::create_child() {
	NN_Link* child_node = new NN_Link;

	child_node->_p_link = this;
	child_node->_forward = _forward;
	child_node->_backward = _backward;
	child_node->trainable = trainable;
	_p_link = child_node;

	return child_node;
}

NN_Link::Layer NN_Link::operator()(NN_Link::Layer prev_node) {
	for (NN_Link::Layer& p_prev_node : prev_node) {
		NN_LinkPtr& m_prev = p_prev_node._val;

		m_prev._p_node->set_link(this, m_prev._n_node);
		_prev.push_back(m_prev._p_node);
	}

	return NN_LinkPtr({ 0, this });
}

void NN_Link::set_link(NN_Link* node, int index) {
	_next.push_back(node);
}