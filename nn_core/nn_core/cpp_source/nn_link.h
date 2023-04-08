#pragma once
#include "nn_base_layer.h"


/**********************************************/
/*                                            */
/*                  Layer_Ptr                 */
/*                                            */
/**********************************************/

template <class _T>
struct Layer_t {
public:
	_T* _link;
	NN_Tensor* _output;
	vector<NN_Tensor*>* _d_output;
};

/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

class NN_Link {
public:
	vector<NN_Link*> _prev;
	vector<NN_Link*> _next;

	NN_Link* _parent;
	vector<NN_Link*> _child;

	vector<NN_Tensor*> _input;
	NN_Tensor _output;

	vector<NN_Tensor*> _d_output;
	NN_Tensor _d_input;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Link* create_child();
	virtual vector<Layer_t<NN_Link>> operator()(vector<Layer_t<NN_Link>>& prev_node);
	virtual vector<Layer_t<NN_Link>> operator()(initializer_list<vector<Layer_t<NN_Link>>> prev_node);

	void operator()(NN_Link* prev_node);
	
	static void set_child_link(NN_Link* current_node);
};

/**********************************************/
/*                                            */
/*                  NN_Create                 */
/*                                            */
/**********************************************/

template <class _T>
NN_Link& NN_Creater(const _T& m_layer) {
	NN_Layer* layer = new _T(m_layer);
	NN_Link* node = new NN_Link();

	node->_forward = layer;

	NN_Manager::add_node(node);
	NN_Manager::add_layer(layer);

	return *node;
}