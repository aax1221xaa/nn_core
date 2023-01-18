#include "nn_tensor.h"


const size_t NN_Tensor::get_elem_size() const {
	size_t size = 1;

	for (int n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[NN_Tensor::get_length] shape is invalid %s",
				shape.get_str()
			);
		}
		size *= n;
	}

	return size;
}

NN_Tensor::NN_Tensor(const int _device_type) :
	device_type(_device_type)
{
	data = NULL;
}

NN_Tensor::NN_Tensor(const NN_Shape& _shape, const int _device_type) :
	shape(_shape),
	device_type(_device_type)
{
	try {
		size_t size = sizeof(float) * get_elem_size();

		if (device_type == CPU) {
			data = new float[size];
		}
		else if (device_type == GPU) {
			check_cuda(cudaMalloc(&data, size));
		}
	}
	catch (Exception& e) {
		e.Put();
	}
}

NN_Tensor::~NN_Tensor() {
	try {
		if (device_type == CPU) delete[] data;
		else if (device_type == GPU) check_cuda(cudaFree(data));
	}
	catch (Exception& e) {
		e.Put();
	}
}

void NN_Tensor::clear() {
	if (device_type == CPU) delete[] data;
	else if (device_type == GPU) check_cuda(cudaFree(data));

	data = NULL;
	shape.clear();
}

void NN_Tensor::create(const NN_Shape& _shape) {
	clear();
	shape = _shape;

	size_t len = get_elem_size();

	if (device_type == CPU) {
		data = new float[len];
	}
	else if (device_type == GPU) {
		check_cuda(cudaMalloc(&data, sizeof(float) * len));
	}
}

void NN_Tensor::copy_to(NN_Tensor& dst) {
	size_t src_len = sizeof(float) * get_elem_size();
	size_t dst_len = sizeof(float) * dst.get_elem_size();

	if (src_len != dst_len) {
		dst.clear();
		dst.create(shape);
	}

	if (device_type == CPU) {
		if (dst.device_type == CPU) {
			memcpy_s(dst.data, src_len, data, src_len);
		}
		else if (dst.device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, src_len, cudaMemcpyHostToDevice));
		}
	}
	else if (device_type == GPU) {
		if (dst.device_type == CPU) {
			check_cuda(cudaMemcpy(dst.data, data, src_len, cudaMemcpyDeviceToHost));
		}
		else if (dst.device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, src_len, cudaMemcpyDeviceToDevice));
		}
	}
}