#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../CppSource/nn_tensor_plus.h"


NN_Shape get_out_shape(
	const NN_Shape& in_shape,
	int amounts,
	const NN_Shape&& f_shape,
	const NN_Shape&& stride,
	const std::string&& pad
	);

void conv2d(
	NN_Stream& st,
	const GpuTensor<nn_type>& src,
	const GpuTensor<nn_type>& weight,
	GpuTensor<nn_type>& dst,
	const std::string&& pad,
	const NN_Shape&& stride
);



#endif // !_CONVOLUTION_CUH_