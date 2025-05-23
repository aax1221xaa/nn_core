#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>



#define check_cuda(status) (__checkCUDA(status, __FILE__, __LINE__))

void __checkCUDA(cudaError_t status, const char *file, int line);


