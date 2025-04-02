#include "nn_lambda.h"


/**********************************************/
/*                                            */
/*				  NN_BwdOperator              */
/*                                            */
/**********************************************/
/*
NN_BwdOperator::NN_BwdOperator() {

}

NN_BwdOperator::~NN_BwdOperator() {

}

void NN_BwdOperator::run(
	NN_Stream& st,
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& doutput,
	GpuTensor<nn_type>& dinput
) {
	ErrorExcept(
		"[NN_BwdOperator::op] Make this function."
	);
}
*/

/**********************************************/
/*                                            */
/*				   NN_Operator                */
/*                                            */
/**********************************************/

NN_Operator::NN_Operator(const std::string& name) :
	NN_Layer(name),
	_status(-1)
{

}

NN_Operator::NN_Operator(const NN_Operator& p) :
	NN_Layer(p),
	_status(p._status)
{

}

NN_Operator::~NN_Operator() {

}

void NN_Operator::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	switch (_status)
	{
	case 3:
		if (input_shape[0].val() != input_shape[1].val()) {
			ErrorExcept(
				"[NN_Operator::get_output_shape] a_shape and b_shape are different. %s != %s",
				shape_to_str(input_shape[0].val()),
				shape_to_str(input_shape[1].val())
			);
		}
	case 0:
	case 1:
	case 2:
		output_shape = input_shape;
		break;
	default:
		break;
	}
}

//void NN_Operator::run(NN_Stream& st, const GpuTensor<nn_type>& a, const GpuTensor<nn_type>& b, GpuTensor<nn_type>& c) {
//	ErrorExcept(
//		"[NN_Operator::op] Make this function."
//	);
//}
//
//NN_BwdOperator* NN_Operator::create_backward() {
//	return NULL;
//}
//
//void NN_Operator::set_output(const NN_Shape& output_shape, GpuTensor<nn_type>& output) {
//	output.resize(output_shape);
//}

void NN_Operator::set_const_value(nn_type val, int status) {
	ErrorExcept(
		"[NN_Operator::set_const_value] Make this function."
	);
}

int NN_Operator::get_status() {
	return _status;
}


/**********************************************/
/*                                            */
/*				   NN_OpLinker                */
/*                                            */
/**********************************************/

NN_OpLinker::NN_OpLinker(NN_Operator* m_operator) :
	_operator(m_operator)
{

}

NN_OpLinker::NN_OpLinker(const NN_OpLinker& p) :
	_operator(p._operator)
{

}

NN_OpLinker::~NN_OpLinker() {

}

NN_Operator* NN_OpLinker::get_operator() {
	return _operator;
}

const NN_Operator* NN_OpLinker::get_operator() const {
	return _operator;
}

NN_OpLinker* NN_OpLinker::create_child() {
	NN_OpLinker* child_node = new NN_OpLinker(_operator);

	return child_node;
}

NN_OpLinker* NN_OpLinker::operator()(NN_OpLinker* prev_a) {
	set_prev_node(prev_a);
	prev_a->set_next_node(this);

	_operator->set_const_value(0.f, 0);

	return this;
}

NN_OpLinker* NN_OpLinker::operator()(NN_OpLinker* prev_a, nn_type val) {
	set_prev_node(prev_a);
	prev_a->set_next_node(this);

	_operator->set_const_value(val, 1);

	return this;
}

NN_OpLinker* NN_OpLinker::operator()(nn_type val, NN_OpLinker* prev_b) {
	set_prev_node(prev_b);
	prev_b->set_next_node(this);

	_operator->set_const_value(val, 2);

	return this;
}

NN_OpLinker* NN_OpLinker::operator()(NN_OpLinker* prev_a, NN_OpLinker* prev_b) {
	set_prev_node(prev_a);
	set_prev_node(prev_b);
	prev_a->set_next_node(this);
	prev_b->set_next_node(this);

	_operator->set_const_value(0.f, 3);

	return this;
}


/**********************************************/
/*                                            */
/*                  NN_Lambda                 */
/*                                            */
/**********************************************/

NN_Lambda::NN_Lambda(NN_Manager& manager, LAMBDA_FUNC fp, const std::string& layer_name) :
	_manager(manager),
	_fp(fp),
	NN_Layer(layer_name),
	_is_set_nodes(false)
{

}

NN_Lambda::NN_Lambda(const NN_Lambda& p) :
	NN_Layer(p),
	_manager(p._manager),
	_fp(p._fp),
	_io_nodes(p._io_nodes),
	_in_nodes(p._in_nodes),
	_out_nodes(p._out_nodes),
	_is_set_nodes(p._is_set_nodes)
{

}

NN_Lambda::~NN_Lambda() {

}

void NN_Lambda::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	if (!_is_set_nodes) {
		for (size_t i = 0; i < input_shape.size(); ++i) {
			NN_OpLinker* in_node = new NN_OpLinker(NULL);

			_in_nodes.push_back(in_node);
			_io_nodes.push_back(in_node);
			_manager.set_nodes(in_node);
		}

		_out_nodes = _fp(*this, _in_nodes);
		_is_set_nodes = true;
	}

	NN_List<NN_Shape>& m_shapes = _manager.get_node_shape();
	bool is_set_shapes = false;

	if (m_shapes.size() == 0) {
		_manager.set_reserved_shapes();
		is_set_shapes = true;
	}

	for (NN_OpLinker* node : _io_nodes) {
		NN_List<NN_Shape>& m_output_shape = m_shapes[node->get_index()];
		NN_List<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				m_input_shape.append(m_shapes[p_prev_node->get_index()]);
			}
			node->get_operator()->get_output_shape(m_input_shape, m_output_shape);
		}
		else {
			size_t i = 0;
			
			for (NN_OpLinker* p_input : _in_nodes) {
				if (p_input == node) break;
				else ++i;
			}
			m_output_shape.append(input_shape[i]);
		}
	}

	for (NN_OpLinker* p_node : _out_nodes) {
		output_shape.append(m_shapes[p_node->get_index()]);
	}

	if(is_set_shapes) _manager.clear_shapes();
}

void NN_Lambda::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	NN_List<GpuTensor<nn_type>>& nodes_output = _manager.get_node_output();

	for (NN_OpLinker* p_node : _io_nodes) {
		const int n_node = p_node->get_index();
		NN_List<GpuTensor<nn_type>>& m_output = nodes_output[n_node];
		NN_List<GpuTensor<nn_type>> m_input;

		if (p_node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : p_node->get_prev_nodes()) {
				const int prev_index = p_prev_node->get_index();

				m_input.append(nodes_output[prev_index]);
			}
		}
		else {
			int i = 0;

			for (NN_OpLinker* in_node : _in_nodes) {
				if (in_node == p_node) break;
				else ++i;
			}

			m_input.append(input[i]);
		}
		
		NN_Operator* p_operator = p_node->get_operator();
		NN_Stream& st = _manager.get_streams();

		p_operator->run(st, m_input, m_output);
	}
}

NN_Backward* NN_Lambda::create_backward(std::vector<bool>& mask) {
	return NULL;
}

void NN_Lambda::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	NN_List<GpuTensor<nn_type>>& nodes_output = _manager.get_node_output();
	NN_List<NN_Shape>& nodes_shape = _manager.get_node_shape();

	for (NN_OpLinker* node : _io_nodes) {
		const int node_id = node->get_index();
		NN_List<GpuTensor<nn_type>>& m_output = nodes_output[node_id];
		NN_List<GpuTensor<nn_type>> m_input;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				m_input.append(nodes_output[p_prev_node->get_index()]);
			}
		}
		else {
			int i = 0;
			
			for (NN_OpLinker* in_node : _in_nodes) {
				if (node == in_node) break;
				else ++i;
			}
			m_input.append(input[i]);
		}
		node->get_operator()->set_output(nodes_shape[node_id], m_input, m_output);
	}
	for(NN_OpLinker* out_node : _out_nodes) output.append(nodes_output[out_node->get_index()]);
}

void NN_Lambda::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}