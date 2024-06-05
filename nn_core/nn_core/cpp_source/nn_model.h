#pragma once
#include "nn_base.h"
#include "nn_sample.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include <time.h>


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

class Model : public NN_Layer, public NN_Link {
private:
	std::vector<NN_Link*> _input_nodes;
	std::vector<NN_Link*> _output_nodes;
	std::vector<NN_Link*> _layers;

	std::vector<int> _output_indice;

	NN_Manager& _manager;

	const NN_Input* get_input_layer(NN_Link* link);
	void find_path(Layer_t& inputs, Layer_t& outputs, std::vector<int>& find_mask);
	void count_branch(std::vector<int>& mask);
	void set_childs(Layer_t& inputs, Layer_t& outputs, std::vector<int>& mask);

protected:
	const std::vector<int>& get_output_indice() const;
	void set_output_indice(const std::vector<int>& indice);

public:
	static int _stack;

	Model(NN_Manager& manager, const char* model_name);
	Model(NN_Manager& manager, Layer_t inputs, Layer_t outputs, const char* model_name);
	~Model();

	Layer_t operator()(Layer_t prev_node);

	/************************** NN_Link **************************/
	NN_Link* create_child();
	void set_next_link(NN_Link* node, int index);
	/*************************************************************/

	/************************** NN_Layer **************************/
	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
	void run_backward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& d_output, std::vector<GpuTensor<nn_type>>& d_input);
	/**************************************************************/

	void summary();

	template <typename _xT, typename _yT>
	std::vector<std::vector<Tensor<nn_type>>> predict(const Sample<_xT, _yT>& sample);

	template <typename _T>
	std::vector<std::vector<Tensor<nn_type>>> predict(const std::vector<Tensor<_T>>& x);
};

template <typename _xT, typename _yT>
std::vector<std::vector<Tensor<nn_type>>> Model::predict(const Sample<_xT, _yT>& sample) {
	_manager.set_reserved_shapes();
	_manager.set_reserved_outputs();

	std::vector<std::vector<Tensor<nn_type>>> outputs(sample.get_iteration());

	for(std::vector<Tensor<nn_type>>& m_tensor : outputs){
		m_tensor.resize(_output_nodes.size());
	}

	clock_t cnt = 0;

	int i = 0;
	
	for (const DataSet<_xT, _yT>& data : sample) {
		if (i == 0) {
			const NN_Shape shape = data._x.get_shape();

			for (NN_Link* node : _layers) {
				std::vector<NN_Shape>& m_output_shape = _manager.get_node_shape(node->get_index());
				std::vector<NN_Shape> m_input_shape;
				std::vector<GpuTensor<nn_type>>& m_output = _manager.get_node_output(node->get_index());

				if (node->get_prev_nodes().size() > 0) {
					for (NN_Link* p_prev_node : node->get_prev_nodes()) {
						size_t j = 0;

						for (NN_Link* p_next_node : p_prev_node->get_next_nodes()) {
							if (p_next_node == node) break;
							else ++j;
						}
						m_input_shape.push_back(_manager.get_node_shape(p_prev_node->get_index())[j]);
					}
				}

				node->get_layer().get_output_shape(m_input_shape, m_output_shape);
				node->get_layer().build(m_input_shape);

				for (NN_Shape& m_shape : m_output_shape) {
					m_shape[0] = shape[0];
					m_output.push_back(GpuTensor<nn_type>(m_shape));
				}
			}
		}

		//std::cout << "Iteration: " << i << std::endl;
		
		const std::vector<NN_Input*>& inputs = _manager.get_input_layers();

		for (NN_Link* node : _layers) {
			std::vector<GpuTensor<nn_type>>& m_output = _manager.get_node_output(node->get_index());
			std::vector<GpuTensor<nn_type>> m_input;

			if (node->get_prev_nodes().size() > 0) {
				for (NN_Link* p_prev_node : node->get_prev_nodes()) {
					int j = 0;

					for (NN_Link* p_next_node : p_prev_node->get_next_nodes()) {
						if (p_next_node == node) break;
						else ++j;
					}

					m_input.push_back(_manager.get_node_output(p_prev_node->get_index())[j]);
				}
				
				clock_t start = clock();
				node->get_layer().run_forward(_manager.get_streams(), m_input, m_output);
				cnt += clock() - start;
			}
			else {
				int j = 0;
				for (const NN_Input* p_input : inputs) {
					if (&(node->get_layer()) == p_input) {
						p_input->trans_data(data._x, m_output[0]);
						break;
					}
					else ++j;
				}
			}
		}
		
		int m = 0;
		for (NN_Link* p_out_link : _output_nodes) {
			const std::vector<GpuTensor<nn_type>>& out_tensor = _manager.get_node_output(p_out_link->get_index());

			for (const GpuTensor<nn_type>& g_out : out_tensor) {
				Tensor<nn_type> h_out(g_out.get_shape());

				h_out = g_out;
				outputs[i][m++] = h_out;
			}
		}
		
		++i;
	}

	std::cout << cnt << "ms" << std::endl;

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	return outputs;
}

template <typename _T>
std::vector<std::vector<Tensor<nn_type>>> Model::predict(const std::vector<Tensor<_T>>& x) {
	_manager.set_reserved_shapes();
	_manager.set_reserved_outputs();

	std::vector<std::vector<Tensor<nn_type>>> outputs(x.size());

	for (std::vector<Tensor<nn_type>>& m_tensor : outputs) {
		m_tensor.resize(_output_nodes.size());
	}

	int i = 0;

	for (const Tensor<_T>& mx : x) {
		if (i == 0) {
			const NN_Shape shape = mx.get_shape();

			for (NN_Link* node : _layers) {
				std::vector<NN_Shape>& m_output_shape = _manager.get_node_shape(node->get_index());
				std::vector<NN_Shape> m_input_shape;
				std::vector<GpuTensor<nn_type>>& m_output = _manager.get_node_output(node->get_index());

				if (node->get_prev_nodes().size() > 0) {
					for (NN_Link* p_prev_node : node->get_prev_nodes()) {
						size_t j = 0;

						for (NN_Link* p_next_node : p_prev_node->get_next_nodes()) {
							if (p_next_node == node) break;
							else ++j;
						}
						m_input_shape.push_back(_manager.get_node_shape(p_prev_node->get_index())[j]);
					}
				}

				node->get_layer().get_output_shape(m_input_shape, m_output_shape);
				node->get_layer().build(m_input_shape);

				for (NN_Shape& m_shape : m_output_shape) {
					m_shape[0] = shape[0];
					m_output.push_back(GpuTensor<nn_type>(m_shape));
				}
			}
		}

		std::cout << "Iteration: " << i << std::endl;
		const std::vector<NN_Input*>& inputs = _manager.get_input_layers();

		for (NN_Link* node : _layers) {
			std::vector<GpuTensor<nn_type>>& m_output = _manager.get_node_output(node->get_index());
			std::vector<GpuTensor<nn_type>> m_input;

			if (node->get_prev_nodes().size() > 0) {
				for (NN_Link* p_prev_node : node->get_prev_nodes()) {
					int j = 0;

					for (NN_Link* p_next_node : p_prev_node->get_next_nodes()) {
						if (p_next_node == node) break;
						else ++j;
					}

					m_input.push_back(_manager.get_node_output(p_prev_node->get_index())[j]);
				}

				node->get_layer().run_forward(_manager.get_streams(), m_input, m_output);
			}
			else {
				int j = 0;
				for (const NN_Input* p_input : inputs) {
					if (&(node->get_layer()) == p_input) {
						p_input->trans_data(mx, m_output[0]);
						break;
					}
					else ++j;
				}
			}
		}

		int m = 0;
		for (NN_Link* p_out_link : _output_nodes) {
			const std::vector<GpuTensor<nn_type>>& out_tensor = _manager.get_node_output(p_out_link->get_index());

			for (const GpuTensor<nn_type>& g_out : out_tensor) {
				Tensor<nn_type> h_out(g_out.get_shape());

				h_out = g_out;
				outputs[i][m++] = h_out;
			}
		}

		++i;
	}

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	return outputs;
}
