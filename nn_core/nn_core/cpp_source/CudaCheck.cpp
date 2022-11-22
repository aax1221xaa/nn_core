#include "CudaCheck.h"



void __checkCUDA(cudaError_t status, const char *file, int line) {
	if (status != CUDA_SUCCESS) {
		__ErrorException(file, line, "[CUDA ERROR] script=%s", cudaGetErrorString(status));
	}
}

void __checkCUDNN(cudnnStatus_t status, const char *file, int line) {
	char message[200] = { '\0', };

	if (status != CUDNN_STATUS_SUCCESS) {
		__ErrorException(file, line, "[CUDNN ERROR] script=%s", cudnnGetErrorString(status));
	}
}