#pragma once
#include "nn_base_layer.h"


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

class NN_Link {
public:
	struct NN_LinkPtr {
		int _n_node;
		NN_Link* _p_node;
	};

	typedef List<NN_LinkPtr> Layer;

	uint _mark;

	NN_Link* _p_link;

	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;

	GpuTensor<nn_type> _output;

	bool trainable;

	NN_Layer* _forward;
	NN_BackPropLayer* _backward;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Link* create_child();
	virtual Layer operator()(Layer prev_node);

	virtual void set_link(NN_Link* node, int index);
};

#define Layer_t	NN_Link::Layer