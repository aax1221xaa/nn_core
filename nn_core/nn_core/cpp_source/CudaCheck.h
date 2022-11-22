#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include "Exception.h"


#define checkCuda(status) (__checkCUDA(status, __FILE__, __LINE__))
#define checkCudnn(status) (__checkCUDNN(status, __FILE__, __LINE__))


void __checkCUDA(cudaError_t status, const char *file, int line);
void __checkCUDNN(cudnnStatus_t status, const char *file, int line);