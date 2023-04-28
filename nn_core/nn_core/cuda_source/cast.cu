#include "../cuda_source/cast.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

//#include <device_functions.h>
#include <device_launch_parameters.h>

#ifdef FIX_MODE

template <typename dT, typename sT>
__global__ void __cast(
	void* dst,
	void* src,
	cuint elem_size
) {
	cuint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < elem_size) ((dT*)dst)[index] = (dT)((sT*)src)[index];
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
void cast_function(cudaStream_t s, void* dst, dtype src_type, void* src, cuint len) {
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, len);

	switch (src_type)
	{
	case dtype::boolean:
		__cast<_T, bool> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::int8:
		__cast<_T, char> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::uint8:
		__cast<_T, unsigned char> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::int32:
		__cast<_T, int> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::uint32:
		__cast<_T, unsigned int> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::int64:
		__cast<_T, long> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::uint64:
		__cast<_T, unsigned long> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::float32:
		__cast<_T, float> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	case dtype::float64:
		__cast<_T, double> << <blocks, threads, 0, s >> > (dst, src, len);
		break;
	default:
		break;
	}

	//check_cuda(cudaStreamSynchronize(s));
	//check_cuda(cudaGetLastError());
}

void type_cast(cudaStream_t s, dtype dst_type, void* dst, dtype src_type, void* src, cuint len) {
	switch (dst_type)
	{
	case dtype::boolean:
		cast_function<bool>(s, dst, src_type, src, len);

		break;
	case dtype::int8:
		cast_function<char>(s, dst, src_type, src, len);

		break;
	case dtype::uint8:
		cast_function<unsigned char>(s, dst, src_type, src, len);

		break;
	case dtype::int32:
		cast_function<int>(s, dst, src_type, src, len);

		break;
	case dtype::uint32:
		cast_function<unsigned int>(s, dst, src_type, src, len);

		break;
	case dtype::int64:
		cast_function<long>(s, dst, src_type, src, len);

		break;
	case dtype::uint64:
		cast_function<unsigned long>(s, dst, src_type, src, len);

		break;
	case dtype::float32:
		cast_function<float>(s, dst, src_type, src, len);

		break;
	case dtype::float64:
		cast_function<double>(s, dst, src_type, src, len);

		break;
	default:
		break;
	}
}

#endif

#ifndef FIX_MODE
template <typename dT, typename sT>
__global__ void __cast(
	void* dst,
	void* src,
	const uint elem_size
) {
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < elem_size) ((dT*)dst)[index] = ((sT*)src)[index];
}

template <typename _T>
void cast_function(
	cudaStream_t s,
	void* dst,
	const int src_type,
	void* src,
	const uint len) 
{
	dim3 block(SQR_BLOCK_SIZE);
	dim3 grid = get_grid_size(block, len);

	switch (src_type)
	{
	case CV_8SC1:
		__cast<_T, char> << <block, grid, 0, s >> > (dst, src, len);
		break;
	case CV_8UC1:
		__cast<_T, uchar> << <block, grid, 0, s >> > (dst, src, len);
		break;
	case CV_16SC1:
		__cast<_T, short> << <block, grid, 0, s >> > (dst, src, len);
		break;
	case CV_16UC1:
		__cast<_T, ushort> << <block, grid, 0, s >> > (dst, src, len);
		break;
	case CV_32SC1:
		__cast<_T, int> << <block, grid, 0, s >> > (dst, src, len);
		break;
	case CV_32FC1:
		__cast<_T, float> << <block, grid, 0, s >> > (dst, src, len);
		break;
	case CV_64FC1:
		__cast<_T, double> << <block, grid, 0, s >> > (dst, src, len);
		break;
	default:
		break;
	}

	check_cuda(cudaStreamSynchronize(s));
}

void type_cast(cudaStream_t s,
	const int dst_type,
	void* dst,
	const int src_type,
	void* src,
	const uint len) 
{
	switch (dst_type)
	{
	case CV_8SC1:
		cast_function<char>(s, dst, src_type, src, len);
		break;
	case CV_8UC1:
		cast_function<uchar>(s, dst, src_type, src, len);
		break;
	case CV_16SC1:
		cast_function<short>(s, dst, src_type, src, len);
		break;
	case CV_16UC1:
		cast_function<ushort>(s, dst, src_type, src, len);
		break;
	case CV_32SC1:
		cast_function<int>(s, dst, src_type, src, len);
		break;
	case CV_32FC1:
		cast_function<float>(s, dst, src_type, src, len);
		break;
	case CV_64FC1:
		cast_function<double>(s, dst, src_type, src, len);
		break;
	default:
		break;
	}
}
#endif