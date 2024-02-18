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
	Model(Layer_t inputs, Layer_t outputs, const char* model_name);
	~Model();

	NN_Link* create_child();
	Layer_t operator()(Layer_t prev_node);

	void set_link(NN_Link* node, int index);

	void calculate_output_size(std::vector<nn_shape>& input_shape, nn_shape& out_shape);
	void build(std::vector<nn_shape>& input_shape);
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);

	void summary();
};