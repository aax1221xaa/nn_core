#pragma once
#include "nn_base.h"
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
	std::vector<Tensor<nn_type>> predict(const std::vector<Tensor<nn_type>>& x);
};