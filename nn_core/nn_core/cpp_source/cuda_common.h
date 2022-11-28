#pragma once
#include "CudaCheck.h"


typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

#define BLOCK_SIZE				32
#define SQR_BLOCK_SIZE			BLOCK_SIZE * BLOCK_SIZE
#define CONST_ELEM_SIZE			16384
#define CONST_MEM_SIZE			65536

#define EPSILON					1e-8


enum{CPU = 1, GPU = 2};

typedef struct __Stream {
	cudaStream_t* str;
	uint str_size;

}Stream;


typedef struct __Tensor {
	float* data;
	int n;
	int c;
	int h;
	int w;

	int type;

}Tensor;

template <class _T>
struct MemBlock {
	_T* data;
	size_t len;
	int type;
};



uint get_elem_size(const Tensor& tensor);
size_t get_mem_size(const Tensor& tensor);
dim3 get_grid_size(dim3 block_size, cuint x = 1, cuint y = 1, cuint z = 1);
void create_streams(Stream& st, cuint amount);
void free_streams(Stream& stream);
void sync_streams(const Stream& stream);
void set_host_tensor(Tensor& tensor, int n, int c, int h, int w);
void set_dev_tensor(Tensor& tensor, int n, int c, int h, int w);
void set_like_tensor(Tensor& dst, const Tensor& src, int mem_type, bool set_zero);
void free_tensor(Tensor& tensor);
void copy_tensor(const Tensor& src, Tensor& dst);
const char* dim_to_str(const Tensor& tensor);
void print_tensor(const Tensor& t);

template<class _T>
void create_host_memblock(MemBlock<_T>& mem, size_t length) {
	mem.data = new _T[length];
	mem.len = length;
	mem.type = CPU;
}

template<class _T>
void create_dev_memblock(MemBlock<_T>& mem, size_t length) {
	check_cuda(cudaMalloc(&mem.data, sizeof(_T) * length));
	mem.len = length;
	mem.type = GPU;
}

template<class _T>
void free_memblock(MemBlock<_T>& mem) {
	if (mem.type == CPU) { 
		delete[] mem.data; 
	}
	else if(mem.type == GPU) {
		check_cuda(cudaFree(mem.data));
	}

	mem.data = NULL;
	mem.len = 0;
}

