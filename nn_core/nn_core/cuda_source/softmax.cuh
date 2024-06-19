#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                  NN_Softmax                */
/*                                            */
/**********************************************/

class NN_Softmax : public NN_Layer {
public:
	NN_Softmax(const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};


#endif