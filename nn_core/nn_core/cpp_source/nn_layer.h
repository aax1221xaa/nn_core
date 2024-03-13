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

	void test(const std::vector<Tensor<nn_type>>& in_val, std::vector<Tensor<nn_type>>& out_val);
};


/**********************************************/
/*                                            */
/*                  NN_Concat                 */
/*                                            */
/**********************************************/

class NN_Concat : public NN_Layer {
public:
	NN_Concat(const char* name);

	void test(const std::vector<Tensor<nn_type>>& in_val, std::vector<Tensor<nn_type>>& out_val);
};