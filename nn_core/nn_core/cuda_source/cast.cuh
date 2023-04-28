#ifndef _CAST_CUH
#define _CAST_CUH

#include "../cpp_source/cuda_common.h"


#ifdef FIX_MODE

enum class dtype { boolean, int8, uint8, int32, uint32, int64, uint64, float32, float64 };

dtype get_type(bool* data);
dtype get_type(char* data);
dtype get_type(unsigned char* data);
dtype get_type(int* data);
dtype get_type(unsigned int* data);
dtype get_type(long* data);
dtype get_type(unsigned long* data);
dtype get_type(float* data);
dtype get_type(double* data);

void type_cast(cudaStream_t s, dtype dst_type, void* dst, dtype src_type, void* src, cuint len);

#endif

#ifndef FIX_MODE

void type_cast(
	cudaStream_t s, 
	const int dst_type, 
	void* dst, 
	const int src_type, 
	void* src, 
	const uint len
);

#endif

#endif // !_CAST_CUH
