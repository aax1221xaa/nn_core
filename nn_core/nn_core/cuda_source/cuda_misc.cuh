#ifndef _CUDA_MISC_CUH_
#define _CUDA_MISC_CUH_

#include "../cpp_source/misc.h"


void transpose(const tensor4d& input_size, const nn_type* input, nn_type* output);


#endif // !_CUDA_MISC_CUH_
