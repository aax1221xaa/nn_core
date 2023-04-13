#pragma once
#include "nn_base_layer.h"


#if !(FIX_MODE)

#define Layer_t vector<Layer_Ptr<NN_Link>>

/**********************************************/
/*                                            */
/*                  Layer_Ptr                 */
/*                                            */
/**********************************************/

template <class _T>
struct Layer_Ptr {
	_T* _link;
	//NN_Tensor* _output;
	//vector<NN_Tensor*>* _d_output;

	int _output_index;
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
	virtual Layer_t operator()(initializer_list<Layer_t> prev_node);
	virtual Layer_t operator()(Layer_t& prev_node);

	virtual int get_node_index(NN_Link* next_node);
	virtual void set_next_node(NN_Link* next_node, int node_index);
	virtual NN_Tensor& get_output(int node_index);
	virtual vector<NN_Tensor*>& get_d_output(int node_index);
	virtual void link_prev_child();

	static NN_Link* get_child(NN_Link* current_parent);
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

#else

#define Layer_t List<Layer_Ptr<NN_Link>>

/**********************************************/
/*                                            */
/*                  Layer_Ptr                 */
/*                                            */
/**********************************************/

template <class _T>
struct Layer_Ptr {
	_T* _link;
	//NN_Tensor* _output;
	//vector<NN_Tensor*>* _d_output;

	int _output_index;
};

/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

class NN_Link {
public:
	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;

	NN_Link* _parent;
	std::vector<NN_Link*> _child;

	std::vector<NN_Tensor*> _input;
	NN_Tensor _output;

	std::vector<NN_Tensor*> _d_output;
	NN_Tensor _d_input;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Link* create_child();
	virtual Layer_t operator()(std::initializer_list<Layer_t> prev_node);
	virtual Layer_t operator()(Layer_t& prev_node);

	virtual int get_node_index(NN_Link* next_node);
	virtual void set_next_node(NN_Link* next_node, int node_index);
	virtual NN_Tensor& get_output(int node_index);
	virtual std::vector<NN_Tensor*>& get_d_output(int node_index);
	virtual void link_prev_child();

	static NN_Link* get_child(NN_Link* current_parent);
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

#endif