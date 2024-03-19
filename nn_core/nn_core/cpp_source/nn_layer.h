#pragma once
#include "nn_base_layer.h"


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
};


/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/

class NN_Test : public NN_Layer {
public:
	NN_Test(const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                  NN_Concat                 */
/*                                            */
/**********************************************/

class NN_Concat : public NN_Layer {
public:
	NN_Concat(const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};