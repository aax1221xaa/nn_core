#include "cuda_kernel.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __func(int* arr, int len) {
	cuint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < len) arr[index] += 10;
}


vector<int> func(vector<int> arr) {
	int* h_arr = new int[arr.size()];
	int* d_arr = NULL;

	check_cuda(cudaMalloc(&d_arr, sizeof(int) * arr.size()));

	for (int i = 0; i < arr.size(); ++i) h_arr[i] = arr[i];
	check_cuda(cudaMemcpy(d_arr, h_arr, sizeof(int) * arr.size(), cudaMemcpyHostToDevice));

	dim3 threads(32);
	dim3 blocks((arr.size() + 32 - 1) / 32);

	__func<<<blocks, threads>>>(d_arr, arr.size());

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaMemcpy(h_arr, d_arr, sizeof(int) * arr.size(), cudaMemcpyDeviceToHost));

	for (int i = 0; i < arr.size(); ++i) arr[i] = h_arr[i];

	delete[] h_arr;
	check_cuda(cudaFree(d_arr));

	return arr;
}