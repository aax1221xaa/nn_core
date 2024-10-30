#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                 NN_Conv2D                  */
/*                                            */
/**********************************************/

class NN_Conv2D : public NN_Layer {
public:
	const int _amounts;
	const NN_Shape _filter_size;
	const NN_Shape _stride;
	const std::string _pad;

	GpuTensor<nn_type> _filter;
	GpuTensor<nn_type> _bias;

	void set_indice(const NN_Tensor4dShape& in, const NN_Filter4dShape& k);

	NN_Conv2D(cuint amounts, const NN_Shape& filter_size, const NN_Shape& stride, const std::string& pad, const std::string& name);

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);

	NN_List<GpuTensor<nn_type>> get_weight();
};

/**********************************************/
/*                                            */
/*                 NN_dConv2D                 */
/*                                            */
/**********************************************/

class NN_dConv2D : public NN_Backward {
	NN_Conv2D& _conv;

public:
	NN_dConv2D(NN_Conv2D& conv);

	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
	NN_Optimizer* create_optimizer(NN_Optimizer& optimizer);
};


#endif // !_CONVOLUTION_CUH_