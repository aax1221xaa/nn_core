#include "../cuda_source/cast.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


template <typename dT, typename sT>
__global__ void __cast(
	sT* src,
	dT* dst,
	size_t elem_size
) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < elem_size) dst[index] = dT(src[index]);
}

dtype get_type(bool* data) {
	return dtype::boolean;
}

dtype get_type(char* data) {
	return dtype::int8;
}

dtype get_type(unsigned char* data) {
	return dtype::uint8;
}

dtype get_type(int* data) {
	return dtype::int32;
}

dtype get_type(unsigned int* data) {
	return dtype::uint32;
}

dtype get_type(long* data) {
	return dtype::int64;
}

dtype get_type(unsigned long* data) {
	return dtype::uint64;
}

dtype get_type(float* data) {
	return dtype::float32;
}

dtype get_type(double* data) {
	return dtype::float64;
}

template <typename _T>
void cast_function(dtype src_type, void* src, _T* dst, size_t len) {
	dim3 threads(BLOCK_1024);
	dim3 blocks((BLOCK_1024 + len - 1) / BLOCK_1024);

	switch (src_type)
	{
	case dtype::boolean:
		__cast<<<blocks, threads>>>((bool*)src, dst, len);
		break;
	case dtype::int8:
		__cast<<<blocks, threads>>>((char*)src, dst, len);
		break;
	case dtype::uint8:
		__cast<<<blocks, threads>>>((unsigned char*)src, dst, len);
		break;
	case dtype::int32:
		__cast<<<blocks, threads>>>((int*)src, dst, len);
		break;
	case dtype::uint32:
		__cast<<<blocks, threads>>>((unsigned int*)src, dst, len);
		break;
	case dtype::int64:
		__cast<<<blocks, threads>>>((long*)src, dst, len);
		break;
	case dtype::uint64:
		__cast<<<blocks, threads>>>((unsigned long*)src, dst, len);
		break;
	case dtype::float32:
		__cast<<<blocks, threads>>>((float*)src, dst, len);
		break;
	case dtype::float64:
		__cast<<<blocks, threads>>>((double*)src, dst, len);
		break;
	default:
		break;
	}

	//check_cuda(cudaStreamSynchronize(s));
	//check_cuda(cudaGetLastError());
}

void type_cast(dtype src_type, void* src, dtype dst_type, void* dst, size_t len) {
	switch (dst_type)
	{
	case dtype::boolean:
		cast_function(src_type, src, (bool*)dst, len);
		break;
	case dtype::int8:
		cast_function(src_type, src, (char*)dst, len);
		break;
	case dtype::uint8:
		cast_function(src_type, src, (unsigned char*)dst, len);
		break;
	case dtype::int32:
		cast_function(src_type, src, (int*)dst, len);
		break;
	case dtype::uint32:
		cast_function(src_type, src, (unsigned int*)dst, len);
		break;
	case dtype::int64:
		cast_function(src_type, src, (long*)dst, len);
		break;
	case dtype::uint64:
		cast_function(src_type, src, (unsigned long*)dst, len);
		break;
	case dtype::float32:
		cast_function(src_type, src, (float*)dst, len);
		break;
	case dtype::float64:
		cast_function(src_type, src, (double*)dst, len);
		break;
	default:
		break;
	}
}