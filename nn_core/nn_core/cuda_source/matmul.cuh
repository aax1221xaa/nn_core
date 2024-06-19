#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "../cpp_source/nn_base.h"



/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

void test_matmul(
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& weight,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
	);

class NN_Dense : public NN_Layer {
public:
	GpuTensor<nn_type> _weight;
	GpuTensor<nn_type> _bias;
	const int _amounts;

	NN_Dense(const int amounts, const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
	std::vector<GpuTensor<nn_type>> get_weight();
};

#endif