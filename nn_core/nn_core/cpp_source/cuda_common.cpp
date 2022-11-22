#include "cuda_common.h"
#include <string.h>



size_t GetTotalSize(const Tensor* dim) {
	return size_t(dim->n * dim->h * dim->w * dim->c) * sizeof(float);
}

int GetBlockSize(int size) {
	return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

Stream* CreateStreams(int amount) {
	Stream* stream = new Stream;

	stream->st = new cudaStream_t[amount];
	stream->st_size = amount;

	for (int i = 0; i < amount; ++i) checkCuda(cudaStreamCreate(&stream->st[i]));

	return stream;
}

void FreeStreams(Stream** stream) {
	for (int i = 0; i < (*stream)->st_size; ++i) {
		checkCuda(cudaStreamDestroy((*stream)->st[i]));
	}
	delete[](*stream)->st;
	delete *stream;
	
	stream = NULL;
}

void SyncStreams(const Stream* stream) {
	for (int i = 0; i < stream->st_size; ++i) checkCuda(cudaStreamSynchronize(stream->st[i]));
}


Tensor* CreateHostTensor(int n, int h, int w, int c) {
	if (n < 1 || h < 1 || w < 1 || c < 1) {
		ErrorExcept("[CreateHostTensor] invalid dimensions: [%d, %d, %d, %d]", n, h, w, c);
	}

	Tensor* tensor = new Tensor;

	size_t size = n * h * w * c;
	
	tensor->data = new float[size];
	tensor->n = n;
	tensor->h = h;
	tensor->w = w;
	tensor->c = c;
	tensor->type = CPU;

	return tensor;
}

Tensor* CreateDeviceTensor(int n, int h, int w, int c) {
	if (n < 1 || h < 1 || w < 1 || c < 1) {
		ErrorExcept("[CreateHostTensor] invalid dimensions: [%d, %d, %d, %d]", n, h, w, c);
	}

	Tensor* tensor = new Tensor;

	size_t size = sizeof(float) * n * h * w * c;

	checkCuda(cudaMalloc(&(tensor->data), size));
	tensor->n = n;
	tensor->h = h;
	tensor->w = w;
	tensor->c = c;
	tensor->type = GPU;

	return tensor;
}

void FreeTensor(Tensor** tensor) {
	Tensor* p_tensor = *tensor;

	if (p_tensor->type == CPU) {
		delete[] p_tensor->data;
	}
	else if (p_tensor->type == GPU) {
		checkCuda(cudaFree(p_tensor->data));
	}

	delete p_tensor;
	*tensor = NULL;
}

void MemCpy(const Tensor* src, Tensor* dst) {
	size_t src_size = GetTotalSize(src);
	size_t dst_size = GetTotalSize(dst);

	if (src_size != dst_size) {
		ErrorExcept("[MemCopy] src와 dst 사이즈가 맞지 않습니다. %d != %d", src_size, dst_size);
	}

	if (src->type == CPU) {
		if (dst->type == CPU) {
			memcpy_s(dst->data, dst_size, src->data, src_size);
		}
		else if (dst->type == GPU) {
			checkCuda(cudaMemcpy(dst->data, src->data, src_size, cudaMemcpyHostToDevice));
		}
	}
	else if (src->type == GPU) {
		if (dst->type == CPU) {
			checkCuda(cudaMemcpy(dst->data, src->data, src_size, cudaMemcpyDeviceToHost));
		}
		else if (dst->type == GPU) {
			checkCuda(cudaMemcpy(dst->data, src->data, src_size, cudaMemcpyDeviceToDevice));
		}
	}
}