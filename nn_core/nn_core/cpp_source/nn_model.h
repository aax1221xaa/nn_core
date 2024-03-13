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

	/************************** NN_Link **************************/
	NN_Link* create_child();
	void set_next_link(NN_Link* node, int index);
	/*************************************************************/

	void find_path(Layer_t& inputs, Layer_t& outputs, std::vector<int>& find_mask);
	void set_childs(Layer_t& inputs, Layer_t& outputs, std::vector<int>& mask);

	const std::vector<int>& get_output_indice();
	void set_output_indice(const std::vector<int>& indice);

	/************************** NN_Layer **************************/
	void test(const std::vector<Tensor<nn_type>>& in_val, std::vector<Tensor<nn_type>>& out_val);
	/**************************************************************/

	void summary();
};