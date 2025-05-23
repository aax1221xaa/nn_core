#ifndef NN_DW_CONV_CUH
#define NN_DW_CONV_CUH

#include "../cpp_source/nn_base.h"

/**********************************************/
/*                                            */
/*                 NN_DwConv2D                */
/*                                            */
/**********************************************/

class NN_DwConv2D : public NN_Layer {
	const NN_Shape _k_size;
	const NN_Shape _stride;
	const std::string _pad;
	const bool _use_bias;

	GpuTensor<nn_type> _kernel;
	GpuTensor<nn_type> _bias;

public:
	NN_DwConv2D(const NN_Shape& k_size, const NN_Shape& stride, const std::string& pad, bool use_bias, const std::string& layer_name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_List<GpuTensor<nn_type>> get_weight();
};


#endif // !NN_DW_CONV_CUH
