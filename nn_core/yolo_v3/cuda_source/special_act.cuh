#ifndef SPECIAL_ACT_CUH
#define SPECIAL_ACT_CUH

#include "../cpp_source/nn_base.h"


class SpecialAct : public NN_Layer {
	const int _n_classes;

public:
	SpecialAct(int n_classes, const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


#endif // !SPECIAL_ACT_CUH
