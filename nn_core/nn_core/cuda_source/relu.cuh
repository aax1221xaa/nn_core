#ifndef RELU_CUH
#define RELU_CUH

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

class NN_ReLU : public NN_Layer {
public:
	NN_ReLU(const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};



#endif // !RELU_CUH
