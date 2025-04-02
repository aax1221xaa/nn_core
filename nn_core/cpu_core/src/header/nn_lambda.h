#pragma once
#include "nn_base.h"


/**********************************************/
/*                                            */
/*				  NN_BwdOperator              */
/*                                            */
/**********************************************/
/*
class NN_BwdOperator : public NN_Backward {
public:
	NN_BwdOperator();
	virtual ~NN_BwdOperator();

	virtual void run(
		NN_Stream& st,
		const GpuTensor<nn_type>& input,
		const GpuTensor<nn_type>& doutput,
		GpuTensor<nn_type>& dinput
	);
};
*/

/**********************************************/
/*                                            */
/*				   NN_Operator                */
/*                                            */
/**********************************************/
/*
	0 = a: in_tensor, b: out_tensor
	1 = a: in_tensor, b: in_scalar, c: out_tensor
	2 = a: in_scalar, b: in_tensor, c: out_tensor
	3 = a: in_tensor, b: in_tensor, c: out_tensor
*/

class NN_Operator : public NN_Layer {
protected:
	int _status;

public:
	NN_Operator(const std::string& name);
	NN_Operator(const NN_Operator& p);
	virtual ~NN_Operator();
	
	// virtual void get_output_shape(const NN_Shape& a_shape, const NN_Shape& b_shape, NN_Shape& c_shape);
	// virtual void run(NN_Stream& st, const GpuTensor<nn_type>& a, const GpuTensor<nn_type>& b, GpuTensor<nn_type>& c);
	// virtual NN_BwdOperator* create_backward();
	// virtual void set_output(const NN_Shape& output_shape, GpuTensor<nn_type>& output);

	virtual void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	virtual void set_const_value(nn_type val, int status);
	int get_status();
};


/**********************************************/
/*                                            */
/*				   NN_OpLinker                */
/*                                            */
/**********************************************/
/*
	0 = a: in_tensor, b: out_tensor
	1 = a: in_tensor, b: in_scalar, c: out_tensor
	2 = a: in_scalar, b: in_tensor, c: out_tensor
	3 = a: in_tensor, b: in_tensor, c: out_tensor
*/

class NN_OpLinker : public NN_Link {
	NN_Operator* _operator;

public:
	NN_OpLinker(NN_Operator* m_operator);
	NN_OpLinker(const NN_OpLinker& p);
	~NN_OpLinker();

	NN_Operator* get_operator();
	const NN_Operator* get_operator() const;

	NN_OpLinker* create_child();
	
	NN_OpLinker* operator()(NN_OpLinker* prev_a);
	NN_OpLinker* operator()(NN_OpLinker* prev_a, nn_type val);
	NN_OpLinker* operator()(nn_type val, NN_OpLinker* prev_b);
	NN_OpLinker* operator()(NN_OpLinker* prev_a, NN_OpLinker* prev_b);
};


/**********************************************/
/*                                            */
/*                  NN_Lambda                 */
/*                                            */
/**********************************************/

class NN_Lambda : public NN_Layer {
public:
	typedef std::vector<NN_OpLinker*>(&LAMBDA_FUNC)(NN_Lambda&, const std::vector<NN_OpLinker*>&);

private:
	NN_Manager& _manager;

	LAMBDA_FUNC _fp;

	std::vector<NN_OpLinker*> _in_nodes;
	std::vector<NN_OpLinker*> _io_nodes;
	std::vector<NN_OpLinker*> _out_nodes;

	bool _is_set_nodes;

public:
	NN_Lambda(NN_Manager& manager, LAMBDA_FUNC fp, const std::string& layer_name);
	NN_Lambda(const NN_Lambda& p);
	~NN_Lambda();

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);

	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);

	template <class _T>
	NN_OpLinker& operator()(const _T& op);
};

template <class _T>
NN_OpLinker& NN_Lambda::operator()(const _T& op) {
	NN_Operator* p_operator = new _T(op);
	NN_OpLinker* p_linker = new NN_OpLinker(p_operator);

	_manager.set_layers(p_operator);
	_manager.set_nodes(p_linker);
	_io_nodes.push_back(p_linker);

	return *p_linker;
}