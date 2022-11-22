#pragma once
#include "CudaCheck.h"

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

#define BLOCK_SIZE		32
#define CONST_SIZE		16384


enum{CPU = 1, GPU = 2};

typedef struct __Stream {
	cudaStream_t* st;
	int st_size;

}Stream;


typedef struct __Tensor {
	float* data;
	int n;
	int h;
	int w;
	int c;

	int type;

}Tensor;



size_t GetTotalSize(const Tensor* dim);
int GetBlockSize(int size);
Stream* CreateStreams(int amount);
void FreeStreams(Stream** stream);
void SyncStreams(const Stream* stream);
Tensor* CreateHostTensor(int n, int h, int w, int c);
Tensor* CreateDeviceTensor(int n, int h, int w, int c);
void FreeTensor(Tensor** tensor);
void MemCpy(const Tensor* src, Tensor* dst);