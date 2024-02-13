#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

class Model : public NN_Layer, public NN_Link {
public:
	/*
	// NN_Layer
	const char* _layer_name;

	// NN_Link
	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;

	NN_Link* _parent;
	std::vector<NN_Link*> _child;

	std::vector<NN_Tensor<nn_type>*> _input;
	NN_Tensor<nn_type> _output;

	std::vector<NN_Tensor<nn_type>*> _d_output;
	std::vector<NN_Tensor<nn_type>*> _d_input;

	std::vector<nn_shape*> _in_shape;
	nn_shape _out_shape;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;
	*/

	std::vector<NN_Link*> _input_nodes;
	std::vector<NN_Link*> _output_nodes;
	std::vector<NN_Link*> _forward_list;
	std::vector<NN_Link*> _backward_list;

	std::vector<int> _output_indices;

	Model(const char* model_name);
	Model(const Layer_t& inputs, const Layer_t& outputs, const char* model_name);
	~Model();

	NN_Link* create_child();
	Layer_t operator()(const Layer_t& prev_node);

	int get_node_index(NN_Link* next_node);
	void set_next_node(NN_Link* next_node, int node_index);
	DeviceTensor<nn_type>& get_output(int node_index);
	std::vector<DeviceTensor<nn_type>*>& get_d_output(int node_index);
	nn_shape& get_out_shape(int node_index);
	void link_prev_child();

	void calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape*>& input_shape);
	void run_forward(cudaStream_t* s, std::vector<DeviceTensor<nn_type>*>& input, DeviceTensor<nn_type>& output);
	void run_backward(cudaStream_t* s, DeviceTensor<nn_type>& d_output, std::vector<DeviceTensor<nn_type>*>& d_input);

	void standby(const std::vector<NN_Loss>& loss, const std::vector<NN_Optimizer>& optimizer);

	template <typename sample_type, typename truth_type>
	std::vector<HostTensor<nn_type>> train_on_batch(const std::vector<HostTensor<sample_type>>& samples, const std::vector<HostTensor<truth_type>>& truth);
	
	template <typename sample_type, typename truth_type>
	std::vector<HostTensor<nn_type>> fit(
		const std::vector<HostTensor<sample_type>>& samples,
		const std::vector<HostTensor<truth_type>>& truth,
		uint batch,
		uint iter
	);

	template <typename d_type>
	std::vector<HostTensor<nn_type>> predict(
		List<HostTensor<d_type>> x,
		const int batch,
		const int steps
	);

	void summary();
};

template <typename sample_type, typename truth_type>
std::vector<HostTensor<nn_type>> Model::train_on_batch(const std::vector<HostTensor<sample_type>>& samples, const std::vector<HostTensor<truth_type>>& truth) {
	return HostTensor<nn_type>();
}

template <typename sample_type, typename truth_type>
std::vector<HostTensor<nn_type>> Model::fit(
	const std::vector<HostTensor<sample_type>>& samples,
	const std::vector<HostTensor<truth_type>>& truth,
	uint batch,
	uint iter
) {
	return HostTensor<nn_type>();
}

template <typename d_type>
std::vector<HostTensor<nn_type>> Model::predict(
	List<HostTensor<d_type>> x,
	const int batch,
	const int steps
) {

	if (x._size != _input_nodes.size()) {
		ErrorExcept(
			"[Model::predict()] %d input layers are different %d samples.",
			_input_nodes.size(), x._size
		);
	}
	
	/*         set inputs         */
	for (int i = 0; i < _input_nodes.size(); ++i) {
		nn_shape* x_shape = new nn_shape;

		x_shape->push_back(batch);
		for (int j = 1; j < x[i].get()._shape.size(); ++j) x_shape->push_back(x[i].get()._shape[j]);

		_input_nodes[i]->_in_shape.push_back(x_shape);
		_input_nodes[i]->_input.push_back(new DeviceTensor<nn_type>(*x_shape));
	}

	/*         set nodes outputs        */
	for (NN_Link* node : _forward_list) {
		node->_forward->calculate_output_size(node->_in_shape, node->_out_shape);
		node->_forward->set_io(node->_out_shape, node->_input, node->_output);
	}

	/*            set outputs           */
	std::vector<HostTensor<nn_type>> output;
	for (NN_Link* node : _output_nodes) {
		nn_shape out_shape;

		out_shape.push_back(batch * steps);
		for (int i = 1; i < node->_out_shape.size(); ++i) out_shape.push_back(node->_out_shape[i]);
		
		output.push_back(HostTensor<nn_type>(out_shape));
	}

	/*           run forward          */
	for (int i = 0; i < steps; ++i) {
		/*        copy samples       */
		for (int j = 0; j < _input_nodes.size(); ++j) {
			cuint data_size = _input_nodes[j]->_output._len;

			if (get_type(x[j].get()._data) != get_type(_input_nodes[j]->_output._data)) {	
				nn_type* p_data = new nn_type[data_size];

				for (uint k = 0; k < data_size; ++k) p_data[k] = (nn_type)(x[j].get()._data[data_size * i + k]);
				check_cuda(cudaMemcpy(_input_nodes[j]->_input[0]->_data, p_data, sizeof(nn_type) * data_size, cudaMemcpyHostToDevice));

				delete[] p_data;
			}
			else check_cuda(cudaMemcpy(_input_nodes[j]->_input[0]->_data, x[j].get()._data + (data_size * i), sizeof(nn_type) * data_size, cudaMemcpyHostToDevice));
		}

		/*       run forward nodes      */
		for (NN_Link* node : _forward_list) {
			node->_forward->run_forward(NN_Manager::_stream, node->_input, node->_output);
		}

		/*        copy to outputs      */
		for (int j = 0; j < _output_nodes.size(); ++j) {
			cuint output_size = _output_nodes[j]->_output._len;
			check_cuda(cudaMemcpy(output[j]._data + (output_size * i), _output_nodes[j]->_output._data, sizeof(nn_type) * output_size, cudaMemcpyDeviceToHost));
		}
	}

	/*        release inputs        */
	for (NN_Link* node : _input_nodes) {
		for (DeviceTensor<nn_type>* tensor : node->_input) delete tensor;
		for (nn_shape* shape : node->_in_shape) delete shape;

		node->_input.clear();
		node->_in_shape.clear();
	}
	
	return output;
}