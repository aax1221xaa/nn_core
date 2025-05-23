#ifndef NN_UPSAMPLE_CUH
#define NN_UPSAMPLE_CUH

#include "../cpp_source/nn_base.h"

/**********************************************/
/*                                            */
/*                NN_UpSample2D               */
/*                                            */
/**********************************************/

class NN_UpSample2D : public NN_Layer {
	const NN_Shape _k_size;

public:
	NN_UpSample2D(const NN_Shape& k_size, const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};

#endif