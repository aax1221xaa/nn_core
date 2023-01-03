#include "nn_tensor.h"


NN_Tensor::NN_Tensor() {
	data = NULL;
	attr = none;
	dtype = none;
	len = 0;
}

NN_Tensor::NN_Tensor(const int _attr, const int _dtype) {
	data = NULL;
	attr = _attr;
	dtype = _dtype;
	len = 0;
}

NN_Tensor::~NN_Tensor() {
	try {
		clear_tensor();
	}
	catch (Exception e) {
		e.Put();
	}
}

void NN_Tensor::clear_tensor() {
	if (attr == CPU) {
		free(data);
	}
	else if (attr == GPU) {
		check_cuda(cudaFree(data));
	}

	data = NULL;
	attr = none;
	dtype = none;
	len = 0;
}

void NN_Tensor::set_tensor(const NN_Shape& shape, const int _attr, const int _dtype) {
	size_t size = 1;

	for (int n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[NN_Tensor::set_tensor] invalid shape %s",
				shape_to_str(shape)
			);
		}
		size *= n;
	}

	if (_attr != attr || _dtype != dtype || len != size) {
		clear_tensor();
	}

	attr = _attr;
	dtype = _dtype;
	len = size;

	size_t elem_size = get_elem_size(dtype);

	if (attr == CPU) {
		data = malloc(len * elem_size);
	}
	else if (attr == GPU) {
		check_cuda(cudaMalloc(&data, len * elem_size));
	}
}

void NN_Tensor::copy_to(NN_Tensor& dst) {
	size_t elem_size = get_elem_size(dtype);

	if (len != dst.len || dtype != dst.dtype) {
		if (dst.attr == CPU) {
			free(dst.data);
			dst.data = malloc(elem_size * len);
		}
		else if (dst.attr == GPU) {
			check_cuda(cudaFree(dst.data));
			check_cuda(cudaMalloc(&dst.data, elem_size * len));
		}
	}

	dst.dtype = dtype;
	dst.len = len;

	if (attr == CPU) {
		if (dst.attr == CPU) {
			memcpy_s(dst.data, elem_size * len, data, elem_size * len);
		}
		else if (dst.attr == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, elem_size * len, cudaMemcpyHostToDevice));
		}
	}
	else if (attr == GPU) {
		if (dst.attr == CPU) {
			check_cuda(cudaMemcpy(dst.data, data, elem_size * len, cudaMemcpyDeviceToHost));
		}
		else if (dst.attr == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, elem_size * len, cudaMemcpyDeviceToDevice));
		}
	}
}

size_t NN_Tensor::get_elem_size(const int type) {
	size_t size = 0;

	switch (type)
	{
	case int8:
	case uint8:
		size = sizeof(char);
		break;

	case int32:
	case uint32:
		size = sizeof(int);
		break;

	case float32:
		size = sizeof(float);
		break;

	case float64:
		size = sizeof(double);
		break;

	default:
		break;
	}

	return size;
}