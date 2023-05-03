#pragma once
#include "nn_manager.h"


#ifdef FIX_MODE

/**********************************************/
/*                                            */
/*                  NN_Create                 */
/*                                            */
/**********************************************/

template <class _T>
NN_Link& NN_Creater(const _T& m_layer) {
	NN_Layer* layer = NULL;
	NN_Link* node = NULL;

	try {
		layer = new _T(m_layer);
		node = new NN_Link();

		if (!NN_Manager::condition) {
			ErrorExcept(
				"[Input()] can't create %s layer.",
				layer->_layer_name
			);
		}

		node->_forward = layer;

		NN_Manager::add_node(node);
		NN_Manager::add_layer(layer);
	}
	catch (const Exception& e) {
		delete layer;
		delete node;

		throw e;
	}

	return *node;
}

/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	nn_shape _shape;

	NN_Input(const nn_shape& input_size, int batch, const char* _layer_name);
	~NN_Input();

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

Layer_t Input(const nn_shape& input_size, int batch, const char* layer_name = "");

/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/
/*
class NN_Test : public NN_Layer {
public:
	NN_Test(const char* name);
	NN_Test(const NN_Test& p);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};
*/
/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

class NN_Dense : public NN_Layer {
public:
	NN_Tensor<nn_type> _weight;
	NN_Tensor<nn_type> _bias;
	const int _amounts;

	NN_Dense(const int amounts, const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

class NN_ReLU : public NN_Layer {
public:
	NN_ReLU(const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

/**********************************************/
/*                                            */
/*                  NN_Conv2D                 */
/*                                            */
/**********************************************/

enum class Pad{VALID, SAME};

class NN_Conv2D : public NN_Layer {
protected:
	static size_t const_offset_cnt;

public:
	size_t const_offset;

	NN_Tensor<nn_type> pad;

	NN_Tensor<nn_type> _kernel;
	NN_Tensor<nn_type> _bias;

	int _amounts;
	const nn_shape _kernel_size;
	const nn_shape _strides;

	Pad _pad;

	bool _do_padding;

	NN_Conv2D(int amounts, const nn_shape& kernel_size , const nn_shape& strides, Pad pad, const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

/**********************************************/
/*                                            */
/*                  NN_Flatten                */
/*                                            */
/**********************************************/

class NN_Flatten : public NN_Layer {
public:
	NN_Flatten(const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void set_io(nn_shape& out_shape, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

/**********************************************/
/*                                            */
/*                 NN_Maxpool2D               */
/*                                            */
/**********************************************/

class NN_Maxpool2D : public NN_Layer {
public:
	const nn_shape _kernel_size;
	const nn_shape _strides;

	NN_Maxpool2D(const nn_shape& kernel_size, const nn_shape& strides, const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t* s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t* s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

#endif

#ifndef FIX_MODE

/**********************************************/
/*                                            */
/*                  NN_Create                 */
/*                                            */
/**********************************************/

template <class _T>
NN_Link& NN_Creater(const _T& m_layer) {
	NN_Layer* layer = NULL;
	NN_Link* node = NULL;

	try {
		layer = new _T(m_layer);
		node = new NN_Link();

		if (!NN_Manager::condition) {
			ErrorExcept(
				"[Input()] can't create %s layer.",
				layer->_layer_name
			);
		}

		node->_forward = layer;

		NN_Manager::add_node(node);
		NN_Manager::add_layer(layer);
	}
	catch (const Exception& e) {
		delete layer;
		delete node;

		throw e;
	}

	return *node;
}

/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	nn_shape _shape;

	NN_Input(const nn_shape& input_size, int batch, const char* _layer_name);
	~NN_Input();

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

Layer_t Input(const nn_shape& input_size, int batch, const char* layer_name = "");

/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/
/*
class NN_Test : public NN_Layer {
public:
	NN_Test(const char* name);
	NN_Test(const NN_Test& p);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};
*/
/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

class NN_Dens : public NN_Layer {
public:
	NN_Tensor<nn_type> _weight;
	NN_Tensor<nn_type> _bias;
	const int _amounts;

	NN_Dens(const int amounts, const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};

/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

class NN_ReLU : public NN_Layer {
public:
	NN_ReLU(const char* name);

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output);
	void run_backward(cudaStream_t s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input);
};


#endif