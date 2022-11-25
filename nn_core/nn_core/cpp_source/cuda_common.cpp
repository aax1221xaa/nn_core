#include "cuda_common.h"
#include <string.h>



uint get_elem_size(const Tensor& tensor) {
	return tensor.n * tensor.c * tensor.h * tensor.w;
}

size_t get_mem_size(const Tensor& tensor) {
	return size_t(tensor.n * tensor.h * tensor.w * tensor.c) * sizeof(float);
}

dim3 get_grid_size(dim3 block_size, cuint x, cuint y, cuint z) {
	dim3 grids;

	grids.x = (x + block_size.x - 1) / block_size.x;
	grids.y = (y + block_size.y - 1) / block_size.y;
	grids.z = (z + block_size.z - 1) / block_size.z;

	return grids;
}

void create_streams(Stream& st, cuint amount) {
	st.str = new cudaStream_t[amount];
	st.str_size = amount;

	for (uint i = 0; i < amount; ++i) check_cuda(cudaStreamCreate(&st.str[i]));
}

void free_streams(Stream& stream) {
	for (uint i = 0; i < stream.str_size; ++i) {
		check_cuda(cudaStreamDestroy(stream.str[i]));
	}

	delete[] stream.str;

	stream.str = NULL;
	stream.str_size = 0;
}

void sync_streams(const Stream& stream) {
	for (uint i = 0; i < stream.str_size; ++i) check_cuda(cudaStreamSynchronize(stream.str[i]));
}


void create_host_tensor(Tensor& tensor, int n, int c, int h, int w) {
	if (n < 1 || h < 1 || w < 1 || c < 1) {
		ErrorExcept("[create_host_tensor] invalid dimensions: [%d, %d, %d, %d]", n, h, w, c);
	}

	size_t size = n * h * w * c;
	
	tensor.data = new float[size];
	tensor.n = n;
	tensor.h = h;
	tensor.w = w;
	tensor.c = c;
	tensor.type = CPU;
}

void create_dev_tensor(Tensor& tensor, int n, int c, int h, int w) {
	if (n < 1 || h < 1 || w < 1 || c < 1) {
		ErrorExcept("[create_dev_tensor] invalid dimensions: [%d, %d, %d, %d]", n, h, w, c);
	}

	size_t size = sizeof(float) * n * h * w * c;

	check_cuda(cudaMalloc(&(tensor.data), size));
	tensor.n = n;
	tensor.h = h;
	tensor.w = w;
	tensor.c = c;
	tensor.type = GPU;
}

void free_tensor(Tensor& tensor) {
	if (tensor.type == CPU) {
		delete[] tensor.data;
	}
	else if (tensor.type == GPU) {
		check_cuda(cudaFree(tensor.data));
	}

	tensor.data = NULL;
	tensor.n = tensor.c = tensor.h = tensor.w = 0;
	tensor.type = 0;
}

void copy_tensor(const Tensor& src, Tensor& dst) {
	size_t src_size = get_mem_size(src);
	size_t dst_size = get_mem_size(dst);

	if (src_size != dst_size) {
		ErrorExcept("[MemCopy] src와 dst 사이즈가 맞지 않습니다. %d != %d", src_size, dst_size);
	}

	if (src.type == CPU) {
		if (dst.type == CPU) {
			memcpy_s(dst.data, dst_size, src.data, src_size);
		}
		else if (dst.type == GPU) {
			check_cuda(cudaMemcpy(dst.data, src.data, src_size, cudaMemcpyHostToDevice));
		}
	}
	else if (src.type == GPU) {
		if (dst.type == CPU) {
			check_cuda(cudaMemcpy(dst.data, src.data, src_size, cudaMemcpyDeviceToHost));
		}
		else if (dst.type == GPU) {
			check_cuda(cudaMemcpy(dst.data, src.data, src_size, cudaMemcpyDeviceToDevice));
		}
	}
}

const char* dim_to_str(const Tensor& tensor) {
	char buffer[100];

	sprintf_s(buffer, "[%d, %d, %d, %d]", tensor.n, tensor.c, tensor.h, tensor.w);

	return buffer;
}