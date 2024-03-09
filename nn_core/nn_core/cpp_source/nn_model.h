#pragma once
#include "nn_base_layer.h"
#include "nn_optimizer.h"
#include "nn_loss.h"


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

public:
	static int _stack;


	Model(NN_Manager& manager, const char* model_name);
	Model(NN_Manager& manager, Layer_t inputs, Layer_t outputs, const char* model_name);
	~Model();

	Layer_t operator()(Layer_t prev_node);
	NN_Link* create_child();
	void set_next_link(NN_Link* node, int index);

	void find_path(Layer_t& inputs, Layer_t& outputs, std::vector<int>& find_mask);
	void set_childs(Layer_t& inputs, Layer_t& outputs, std::vector<int>& mask);

	const std::vector<int>& get_output_indice();
	void set_output_indice(const std::vector<int>& indice);

	nn_shape calculate_output_size(nn_shape& input_shape);
	void build();
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);

	std::vector<int> digestion(const std::vector<int>& feed);

	void summary();
};