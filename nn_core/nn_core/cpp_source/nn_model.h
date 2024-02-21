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
	std::vector<NN_Link*> _input_nodes;
	std::vector<NN_Link*> _output_nodes;
	std::vector<NN_Link*> _layers;
	
	std::vector<int> _out_indice;

	Model(const char* model_name);
	Model(NN_Link::Layer inputs, NN_Link::Layer outputs, const char* model_name);
	~Model();

	NN_Link* create_child();
	NN_Link::Layer operator()(NN_Link::Layer prev_node);

	void set_link(NN_Link* node, int index);

	nn_shape calculate_output_size(nn_shape& input_shape);
	void build();
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);

	void summary();
};