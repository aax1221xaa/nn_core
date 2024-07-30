#pragma once
#include "nn_base.h"
#include "nn_sample.h"
#include "../cuda_source/optimizer.cuh"
#include "nn_loss.h"
#include <time.h>
#include <H5Cpp.h>


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

class Model : public NN_Layer, public NN_Link {
private:
	/******************** NN_Layer ******************/
	// const char* _layer 
	
	/******************* NN_Link ********************/
	// int _index
	// std::vector<NN_Link*> _prev;
	// std::vector<NN_Link*> _next;

	// NN_Layer* _layer;
	// NN_Backward* _backward;

	// std::vector<GpuTensor<nn_type>> _weights;

	// bool trainable

	/***********************************************/

	std::vector<NN_Link*> _input_nodes;
	std::vector<NN_Link*> _output_nodes;
	std::vector<NN_Link*> _layers;

	std::vector<NN_Loss> _losses;
	NN_Optimizer _optimizer;

	std::vector<int> _output_indice;

	NN_Manager& _manager;

	const NN_Input* get_input_layer(NN_Link* link);
	void find_path(Layer_t& inputs, Layer_t& outputs, std::vector<int>& find_mask);
	void count_branch(std::vector<int>& mask);
	void set_childs(Layer_t& inputs, Layer_t& outputs, std::vector<int>& mask);

	static int get_n_node_prev_for_next(const NN_Link* prev_node, const NN_Link* curr_node);
	static int get_n_input(const std::vector<NN_Link*>& input_node, const NN_Link* curr_node);

	const std::vector<int>& get_output_indice() const;
	void set_output_indice(const std::vector<int>& indice);

	static std::vector<std::string> get_layer_names(const H5::H5File& fp);
	static void set_weight(const H5::Group& group, NN_List<GpuTensor<nn_type>>& g_tensor);

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
	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
	NN_List<GpuTensor<nn_type>> get_weight();
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	/**************************************************************/

	void load_weights(const std::string& path, bool skip_mismatch = false);
	void summary();

	template <typename _T>
	NN_List<Tensor<nn_type>> predict(const std::vector<Tensor<_T>>& x, int batch_size, int steps);

	void stand_by(NN_Optimizer& optimizer, std::initializer_list<NN_Loss>& loss);

	template <typename _xT, typename _yT>
	NN_List<Tensor<nn_type>> fit(const DataSet<_xT, _yT>& samples, int batch, int steps);
};

template <typename _T>
NN_List<Tensor<nn_type>> Model::predict(const std::vector<Tensor<_T>>& x, int batch_size, int steps) {
	_manager.set_reserved_shapes();
	_manager.set_reserved_outputs();

	NN_List<NN_Shape>& nodes_shapes = _manager.get_node_shape();
	NN_List<GpuTensor<nn_type>>& nodes_outputs = _manager.get_node_output();
	NN_List<Tensor<nn_type>> outputs;
	std::vector<Tensor<_T>> batch_x(x.size());

	outputs.reserve(steps);
	
	for (NN_List<Tensor<nn_type>>& m_output : outputs) m_output.resize(x.size());

	for (int i = 0; i < steps; ++i) {
		if (i == 0) {
			for (size_t j = 0; j < x.size(); ++j) {
				NN_Shape x_shape = x[j].get_shape();

				x_shape[0] = batch_size;
				batch_x[j].resize(x_shape);
			}

			for (NN_Link* node : _layers) {
				NN_List<NN_Shape>& m_output_shape = nodes_shapes[node->get_index()];
				NN_List<NN_Shape> m_input_shape;
				NN_List<GpuTensor<nn_type>>& m_output = nodes_outputs[node->get_index()];
				NN_List<GpuTensor<nn_type>> m_input;

				if (node->get_prev_nodes().size() > 0) {
					for (NN_Link* p_prev_node : node->get_prev_nodes()) {
						int n_out = get_n_node_prev_for_next(p_prev_node, node);
						int n_prev = p_prev_node->get_index();

						m_input_shape.append(nodes_shapes[n_prev][n_out]);
						m_input.append(nodes_outputs[n_prev][n_out]);
					}
				}
				else {
					int n_input_node = get_n_input(_input_nodes, node);
					NN_Shape x_shape = x[n_input_node].get_shape();

					x_shape[0] = batch_size;
					m_input_shape.append(x_shape);
				}

				node->get_layer().get_output_shape(m_input_shape, m_output_shape);
				//node->get_layer().build(m_input_shape, node);
				node->get_layer().set_output(m_output_shape, m_input, m_output);
			}
		}

		int batch_start = i * batch_size;

		for (int j = 0; j < x.size(); ++j) {
			NN_Shape x_shape = x[j].get_shape();
			int amounts = x_shape[0];
			std::vector<int> indice(batch_size);

			for (int n = 0; n < batch_size; ++n) indice[n] = (batch_start + n) % amounts;

			batch_x[j] = x[j](indice);
		}

		// std::cout << "Iteration: " << i << std::endl;
		const std::vector<NN_Input*>& inputs = _manager.get_input_layers();

		for (NN_Link* node : _layers) {
			NN_List<GpuTensor<nn_type>>& m_output = nodes_outputs[node->get_index()];
			NN_List<GpuTensor<nn_type>> m_input;

			if (node->get_prev_nodes().size() > 0) {
				for (NN_Link* p_prev_node : node->get_prev_nodes()) {
					int n_out = get_n_node_prev_for_next(p_prev_node, node);
					int n_prev = p_prev_node->get_index();

					m_input.append(nodes_outputs[n_prev][n_out]);
				}

				node->get_layer().run(_manager.get_streams(), m_input, m_output);
			}
			else {
				int n_input_node = get_n_input(_input_nodes, node);

				for (const NN_Input* p_input : inputs) {
					if (&(node->get_layer()) == p_input) {
						p_input->trans_data(batch_x[n_input_node], m_output[0].val());
						break;
					}
				}
			}
		}

		int n = 0;
		for (NN_Link* p_out_link : _output_nodes) {
			const GpuTensor<nn_type>& out_tensor = nodes_outputs[p_out_link->get_index()][0].val();

			outputs[i][n].val().resize(out_tensor.get_shape());
			outputs[i][n++].val() = out_tensor;
		}
	}

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	_manager.clear_outputs();
	_manager.clear_shapes();

	return outputs;
}


template <typename _xT, typename _yT>
NN_List<Tensor<nn_type>> Model::fit(const DataSet<_xT, _yT>& samples, int batch, int steps) {
	_manager.set_reserved_shapes();
	_manager.set_reserved_outputs();
	_manager.set_reserved_dinputs();

	NN_List<NN_Shape>& nodes_shapes = _manager.get_node_shape();
	NN_List<GpuTensor<nn_type>>& nodes_outputs = _manager.get_node_output();
	NN_List<GpuTensor<nn_type>>& nodes_doutputs = _manager.get_node_dinput();

	for (int i = 0; i < steps; ++i) {
		std::vector<Tensor<nn_type>> batch_x;
		std::vector<Tensor<nn_type>> batch_y;

		for (const Tensor<nn_type>& sample_x : samples._x) {
			NN_Shape x_shape = sample_x.get_shape();
			const int amounts = x_shape[0];
			std::vector<int> batch_indice(batch, 0);

			for (int j = 0; j < batch; ++j) {
				batch_indice[j] = (i * batch + j) % amounts;
			}

			x_shape[0] = batch;

			Tensor<nn_type> x(x_shape);

			x = sample_x(batch_indice);
			batch_x.push_back(x);
		}

		for (const Tensor<nn_type>& sample_y : samples._y) {
			NN_Shape y_shape = sample_y.get_shape();
			std::vector<int> batch_indice(batch, 0);
			const int amounts = y_shape[0];

			for (int j = 0; j < batch; ++j) batch_indice[j] = (i * batch + j) % amounts;

			y_shape[0] = batch;

			Tensor<nn_type> y(y_shape);

			y = sample_y(batch_indice);
			batch_y.push_back(y);
		}

		if (i == 0) {
			for (NN_Link* node : _layers) {
				NN_List<NN_Shape>& m_output_shape = nodes_shapes[node->get_index()];
				NN_List<NN_Shape> m_input_shape;
				NN_List<GpuTensor<nn_type>>& m_output = nodes_outputs[node->get_index()];
				NN_List<GpuTensor<nn_type>> m_input;

				NN_List<GpuTensor<nn_type>>& m_doutput = nodes_doutputs[node->get_index()];

				if (node->get_prev_nodes().size() > 0) {
					for (NN_Link* p_prev_node : node->get_prev_nodes()) {
						int n_out = get_n_node_prev_for_next(p_prev_node, node);
						int n_prev = p_prev_node->get_index();

						m_input_shape.append(nodes_shapes[n_prev][n_out]);
						m_input.append(nodes_outputs[n_prev][n_out]);
					}
				}
				else {
					int n_input_node = get_n_input(_input_nodes, node);
					NN_Shape x_shape = x[n_input_node].get_shape();

					x_shape[0] = batch_size;
					m_input_shape.append(x_shape);
				}

				node->get_layer().get_output_shape(m_input_shape, m_output_shape);
				//node->get_layer().build(m_input_shape, node);
				node->get_layer().set_output(m_output_shape, m_input, m_output);

				if (node->get_backward()) {

				}
			}
		}

		int batch_start = i * batch_size;

		for (int j = 0; j < x.size(); ++j) {
			NN_Shape x_shape = x[j].get_shape();
			int amounts = x_shape[0];
			std::vector<int> indice(batch_size);

			for (int n = 0; n < batch_size; ++n) indice[n] = (batch_start + n) % amounts;

			batch_x[j] = x[j](indice);
		}

		// std::cout << "Iteration: " << i << std::endl;
		const std::vector<NN_Input*>& inputs = _manager.get_input_layers();

		for (NN_Link* node : _layers) {
			NN_List<GpuTensor<nn_type>>& m_output = nodes_outputs[node->get_index()];
			NN_List<GpuTensor<nn_type>> m_input;

			if (node->get_prev_nodes().size() > 0) {
				for (NN_Link* p_prev_node : node->get_prev_nodes()) {
					int n_out = get_n_node_prev_for_next(p_prev_node, node);
					int n_prev = p_prev_node->get_index();

					m_input.append(nodes_outputs[n_prev][n_out]);
				}

				node->get_layer().run(_manager.get_streams(), m_input, m_output);
			}
			else {
				int n_input_node = get_n_input(_input_nodes, node);

				for (const NN_Input* p_input : inputs) {
					if (&(node->get_layer()) == p_input) {
						p_input->trans_data(batch_x[n_input_node], m_output[0].val());
						break;
					}
				}
			}
		}

		int n = 0;
		for (NN_Link* p_out_link : _output_nodes) {
			const GpuTensor<nn_type>& out_tensor = nodes_outputs[p_out_link->get_index()][0].val();

			outputs[i][n].val().resize(out_tensor.get_shape());
			outputs[i][n++].val() = out_tensor;
		}
	}

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	_manager.clear_outputs();
	_manager.clear_shapes();

	return outputs;
}


/**********************************************/
/*                                            */
/*                    dModel                  */
/*                                            */
/**********************************************/

class dModel : public NN_Backward {
public:
	Model& _model;

	dModel(Model& model);

	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};