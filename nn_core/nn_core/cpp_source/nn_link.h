#pragma once
#include "nn_base_layer.h"

#ifdef FIX_MODE

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

	std::vector<NN_Tensor<nn_type>*> _input;
	NN_Tensor<nn_type> _output;

	std::vector<NN_Tensor<nn_type>*> _d_inputs;
	//NN_Tensor<nn_type> _d_output;
	std::vector<NN_Tensor<nn_type>*> _d_outputs;

	std::vector<nn_shape*> _in_shape;
	nn_shape _out_shape;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Backward* _backward;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Link* create_child();
	virtual Layer_t operator()(const Layer_t& prev_node);

	virtual int get_node_index(NN_Link* next_node);
	virtual void set_next_node(NN_Link* next_node, int node_index);
	virtual NN_Tensor<nn_type>& get_output(int node_index);
	virtual std::vector<NN_Tensor<nn_type>*>& get_d_output(int node_index);
	virtual nn_shape& get_out_shape(int node_index);
	virtual void link_prev_child();

	static NN_Link* get_child(NN_Link* current_parent);
};

#endif

#ifndef FIX_MODE

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

	std::vector<NN_Tensor<nn_type>*> _input;
	NN_Tensor<nn_type> _output;

	std::vector<NN_Tensor<nn_type>*> _d_inputs;
	NN_Tensor<nn_type> _d_output;
	std::vector<NN_Tensor<nn_type>*> _d_outputs;

	std::vector<nn_shape*> _in_shape;
	nn_shape _out_shape;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Link* create_child();
	virtual Layer_t operator()(const Layer_t& prev_node);

	virtual int get_node_index(NN_Link* next_node);
	virtual void set_next_node(NN_Link* next_node, int node_index);
	virtual NN_Tensor<nn_type>& get_output(int node_index);
	virtual std::vector<NN_Tensor<nn_type>*>& get_d_output(int node_index);
	virtual nn_shape& get_out_shape(int node_index);
	virtual void link_prev_child();

	static NN_Link* get_child(NN_Link* current_parent);
};

#endif