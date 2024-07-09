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
	void build(const NN_List<NN_Shape>& input_shape, NN_Link* p_node);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(NN_Optimizer* optimizer);
	NN_List<GpuTensor<nn_type>> get_weight();
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	/**************************************************************/

	void load_weights(const std::string& path, bool skip_mismatch = false);
	void summary();

	template <typename _xT, typename _yT>
	NN_List<Tensor<nn_type>> evaluate(const Sample<_xT, _yT>& sample);

	template <typename _T>
	NN_List<Tensor<nn_type>> predict(const NN_List<Tensor<_T>>& x);

	void stand_by(NN_Optimizer& optimizer, std::initializer_list<NN_Loss>& loss);

	template <typename _xT, typename _yT>
	NN_List<Tensor<nn_type>> fit(const DataSet<_xT, _yT>& sample, int batch);
};

template <typename _xT, typename _yT>
NN_List<Tensor<nn_type>> Model::evaluate(const Sample<_xT, _yT>& sample) {
	_manager.set_reserved_shapes();
	_manager.set_reserved_outputs();

	NN_List<NN_Shape>& nodes_shapes = _manager.get_node_shape();
	NN_List<GpuTensor<nn_type>>& nodes_outputs = _manager.get_node_output();
	NN_List<Tensor<nn_type>> outputs;
	
	outputs.reserve(sample.get_iteration());

	for(NN_List<Tensor<nn_type>>& m_tensor : outputs){
		m_tensor.resize(_output_nodes.size());
	}

	int i = 0;
	
	for (const DataSet<_xT, _yT>& data : sample) {
		Tensor<_xT> data_x = Tensor<_xT>::expand_dims(data._x[0], 1);

		if (i == 0) {
			const NN_Shape shape = data_x.get_shape();

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
					}
				}
				else {
					int n_input_node = get_n_input(_input_nodes, node);

					m_input_shape.append(data_x.get_shape());
				}

				node->get_layer().get_output_shape(m_input_shape, m_output_shape);
				//node->get_layer().build(m_input_shape);
				node->get_layer().set_output(m_output_shape, m_input, m_output);
			}
		}
		
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
				int j = 0;
				for (const NN_Input* p_input : inputs) {
					if (&(node->get_layer()) == p_input) {
						p_input->trans_data(data_x, m_output[0].val());
						break;
					}
					else ++j;
				}
			}
		}
		
		int m = 0;
		for (NN_Link* p_out_link : _output_nodes) {
			const NN_List<GpuTensor<nn_type>>& out_tensor = nodes_outputs[p_out_link->get_index()];

			for (const NN_List<GpuTensor<nn_type>>& g_out : out_tensor) {
				outputs[i][m].val().resize(g_out.val().get_shape());
				outputs[i][m++].val() = g_out.val();
			}
		}
		
		++i;
	}

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	_manager.clear_outputs();
	_manager.clear_shapes();

	return outputs;
}

template <typename _T>
NN_List<Tensor<nn_type>> Model::predict(const NN_List<Tensor<_T>>& x) {
	_manager.set_reserved_shapes();
	_manager.set_reserved_outputs();

	NN_List<NN_Shape>& nodes_shapes = _manager.get_node_shape();
	NN_List<GpuTensor<nn_type>>& nodes_outputs = _manager.get_node_output();
	NN_List<Tensor<nn_type>> outputs;

	outputs.resize(_output_nodes.size());

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

			m_input_shape.append(x[n_input_node].val().get_shape());
		}

		node->get_layer().get_output_shape(m_input_shape, m_output_shape);
		node->get_layer().build(m_input_shape);
		node->get_layer().set_output(m_output_shape, m_input, m_output);
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
					p_input->trans_data(x[n_input_node].val(), m_output[0].val());
					break;
				}
			}
		}
	}

	int n = 0;
	for (NN_Link* p_out_link : _output_nodes) {
		const GpuTensor<nn_type>& out_tensor = nodes_outputs[p_out_link->get_index()][0].val();

		outputs[n].val().resize(out_tensor.get_shape());
		outputs[n++].val() = out_tensor;
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
	Model* _model;

	dModel(Model* model, NN_Optimizer* optimizer);

	void get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape);
	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};