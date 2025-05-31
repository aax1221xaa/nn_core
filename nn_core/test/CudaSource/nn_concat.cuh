#ifndef NN_CONCAT_CUH
#define NN_CONCAT_CUH

#include "../CppSource/nn_tensor_plus.h"


void concat_test(
	NN_Stream& stream,
	const NN_List<GpuTensor<nn_type>>& src,
	GpuTensor<nn_type>& dst,
	cuint axis
);


#endif