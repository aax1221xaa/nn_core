#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STEAM


#include <cuda_runtime.h>
#include <cuda.h>



#define check_cuda(status) (__checkCUDA(status, __FILE__, __LINE__))

void __checkCUDA(cudaError_t status, const char *file, int line);


