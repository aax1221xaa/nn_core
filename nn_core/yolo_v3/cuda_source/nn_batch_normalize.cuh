#ifndef NN_BATCH_NORMALIZE
#define NN_BATCH_NORMALIZE

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*				 NN_BatchNormalize            */
/*                                            */
/**********************************************/

class NN_BatchNormalize : public NN_Layer {
	GpuTensor<nn_type> _means;
	GpuTensor<nn_type> _var;
	GpuTensor<nn_type> _beta;
	GpuTensor<nn_type> _gamma;

public:
	NN_BatchNormalize(const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_List<GpuTensor<nn_type>> get_weight();
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};

#endif