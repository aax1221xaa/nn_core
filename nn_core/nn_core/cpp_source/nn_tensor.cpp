#include "nn_tensor.h"
#include <random>



NN_Tensor::NN_Tensor() {
	id = NULL;
	device_type = 0;
	bytes = 0;
}

NN_Tensor::NN_Tensor(const vector<int>& _shape, int _device_type) {
	try {
		size_t elem_size = _shape.size();

		bytes = sizeof(float) * elem_size;
		device_type = _device_type;

		if (device_type == GPU) check_cuda(cudaMalloc(&data, bytes));
		else if (device_type == CPU) data = new float[elem_size];

		id = linker.Create();
	}
	catch (Exception& e) {
		e.Put();
	}
}

NN_Tensor::NN_Tensor(const size_t _bytes, int _device_type) {
	device_type = _device_type;
	bytes = _bytes;

	try {
		if (_device_type == GPU) check_cuda(cudaMalloc(&data, bytes));
		else if (_device_type == CPU) data = (float*)malloc(bytes);

	}
	catch (Exception& e) {
		e.Put();
	}

	id = linker.Create();
}

NN_Tensor::NN_Tensor(const NN_Tensor& p) {
	id = p.id;

	device_type = p.device_type;
	bytes = p.bytes;
	data = p.data;

	if (id) ++id->ref_cnt;
}

NN_Tensor::~NN_Tensor() {
	clear();
}

const NN_Tensor& NN_Tensor::operator=(const NN_Tensor& p) {
	if (this == &p) return *this;

	clear();

	id = p.id;
	
	device_type = p.device_type;
	bytes = p.bytes;
	data = p.data;

	if (id) ++id->ref_cnt;

	return *this;
}

void NN_Tensor::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			if (device_type == CPU) delete[] data;
			else if (device_type == GPU) check_cuda(cudaFree(data));

			linker.Erase(id);
		}
	}

	device_type = 0;
	data = NULL;
	bytes = 0;

	id = NULL;
}

void NN_Tensor::set(const vector<int>& _shape, int _device_type) {
	clear();

	size_t elem_size = _shape.size();
	bytes = sizeof(float) * elem_size;
	device_type = _device_type;

	if (device_type == GPU) check_cuda(cudaMalloc(&data, sizeof(float) * elem_size));
	else if (device_type == CPU) data = new float[bytes];

	id = linker.Create();
}

void NN_Tensor::copy_to(NN_Tensor& dst, const int _device_type) {
	if (bytes != dst.bytes || device_type != dst.device_type) {
		dst = NN_Tensor(bytes, _device_type);
	}

	if (device_type == CPU) {
		if (_device_type == CPU) {
			memcpy_s(dst.data, dst.bytes, data, bytes);
		}
		else if (_device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, dst.bytes, cudaMemcpyHostToDevice));
		}
	}
	else if (device_type == GPU) {
		if (_device_type == CPU) {
			check_cuda(cudaMemcpy(dst.data, data, dst.bytes, cudaMemcpyDeviceToHost));
		}
		else if (_device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, dst.bytes, cudaMemcpyDeviceToDevice));
		}
	}
}

NN_Tensor NN_Tensor::zeros(const vector<int>& _shape, int _device_type) {
	NN_Tensor p(_shape, _device_type);
	
	if (_device_type == CPU) memset(p.data, 0, p.bytes);
	else if (_device_type == GPU) check_cuda(cudaMemset(p.data, 0, p.bytes));

	return p;
}

NN_Tensor NN_Tensor::zeros_like(const NN_Tensor& p, int _device_type) {
	NN_Tensor p_tensor(p.bytes, _device_type);

	if (_device_type == CPU) memset(p_tensor.data, 0, p_tensor.bytes);
	else if (_device_type == GPU) check_cuda(cudaMemset(p_tensor.data, 0, p_tensor.bytes));

	return p_tensor;
}

void set_uniform(NN_Tensor& p) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(0.f, 1.f);

	size_t size = p.bytes / sizeof(float);

	if (p.device_type == CPU) {
		for (size_t i = 0; i < size; ++i) p.data[i] = dis(gen);
	}
	else if (p.device_type == GPU) {
		float* tmp = new float[size];

		for (size_t i = 0; i < size; ++i) tmp[i] = dis(gen);
		check_cuda(cudaMemcpy(p.data, tmp, sizeof(float) * size, cudaMemcpyHostToDevice));
		
		delete[] tmp;
	}
}