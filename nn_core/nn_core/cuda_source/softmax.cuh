#ifndef __SOFTMAX_CUH__
#define __SOFTMAX_CUH__

#include "../cpp_source/cuda_common.h"


/**********************************************

					softmax

**********************************************/

void softmax(
	cudaStream_t* s,
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape,
	std::vector<uint>& axis
);

#endif // !__SOFTMAX_CUH__
