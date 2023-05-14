#ifndef _CUDA_INDICE_CUH_
#define _CUDA_INDICE_CUH_

#include "../cpp_source/cuda_common.h"


void set_indice(const uint* indice, const size_t size, const size_t offset);
void get_indice(uint* indice, size_t size, size_t offset);
const uint* get_indice_ptr();


#endif 
