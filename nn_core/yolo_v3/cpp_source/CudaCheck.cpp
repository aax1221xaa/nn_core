#include "CudaCheck.h"
#include "Exception.h"


void __checkCUDA(cudaError_t status, const char *file, int line) {
	if (status != CUDA_SUCCESS) {
		__ErrorException(file, line, "[CUDA ERROR] script=%s", cudaGetErrorString(status));
	}
}