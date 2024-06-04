#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "cuda_indice.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

__constant__ uint __indice[CONST_ELEM_SIZE];


void set_indice(const uint* indice, const size_t size, const size_t offset) {
	check_cuda(cudaMemcpyToSymbol(__indice, indice, size, offset));
}

void get_indice(uint* indice, size_t size, size_t offset) {
	check_cuda(cudaMemcpyFromSymbol(indice, __indice, size, offset));
}

uint* get_indice_ptr() {
	uint* symbol = NULL;

	check_cuda(cudaGetSymbolAddress((void**)&symbol, __indice));

	return symbol;
}