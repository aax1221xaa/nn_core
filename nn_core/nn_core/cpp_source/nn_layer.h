#pragma once
#include "nn_manager.h"


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

	nn_shape calculate_output_size(nn_shape& input_shape);
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);
};

Layer_t Input(const nn_shape& input_size, int batch, const char* layer_name);

/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

class NN_Dense : public NN_Layer {
public:
	GpuTensor<nn_type> _weight;
	GpuTensor<nn_type> _bias;
	const int _amounts;

	NN_Dense(const int amounts, const char* name);

	nn_shape calculate_output_size(nn_shape& input_shape);
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);
};
