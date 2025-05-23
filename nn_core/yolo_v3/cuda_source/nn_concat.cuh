#ifndef NN_CONCAT_CUH
#define NN_CONCAT_CUH

#include "../cpp_source/nn_base.h"


void concat_test(
	NN_Stream& stream,
	const NN_List<GpuTensor<nn_type>>& src,
	GpuTensor<nn_type>& dst,
	cuint axis
);

/**********************************************/
/*                                            */
/*                  NN_Concat                 */
/*                                            */
/**********************************************/

class NN_Concat : public NN_Layer {
	int _axis;

public:
	NN_Concat(int axis, const std::string& name = "");
	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};

#endif