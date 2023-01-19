#include "nn_tensor.h"
#include <random>


extern char str_buffer[STR_MAX];
extern int str_idx;


const char* dim_to_str(const NN_Tensor4D& tensor) {
	char tmp_buff[128] = { '\0', };
	char elem[16] = { '\0', };
	int shape[4] = { tensor.n, tensor.c, tensor.h, tensor.w };

	sprintf_s(tmp_buff, "[");

	for (int i = 0; i < 4; ++i) {
		sprintf_s(elem, "%d, ", shape[i]);
		strcat_s(tmp_buff, elem);
	}

	strcat_s(tmp_buff, "]");

	int str_size = strlen(tmp_buff) + 1;
	int clearance = STR_MAX - str_idx;
	char* p_buff = NULL;

	if (clearance >= str_size) {
		p_buff = &str_buffer[str_idx];
		str_idx += str_size;
	}
	else {
		p_buff = str_buffer;
		str_idx = 0;
	}

	strcpy_s(p_buff, sizeof(char) * str_size, tmp_buff);

	return p_buff;
}

size_t get_elem_size(const NN_Tensor4D& tensor) {
	return tensor.n * tensor.c * tensor.h * tensor.w;
}

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

NN_Tensor4D NN_Tensor::get_4dtensor() {
	NN_Tensor4D tensor;
	int* p_dim[4] = { &tensor.n, &tensor.c, &tensor.h, &tensor.w };

	tensor.data = data;
	tensor.n = tensor.c = tensor.h = tensor.w = 1;

	int i = 0;
	for (int dim : shape) {
		*(p_dim[i++]) = dim;
	}

	return tensor;
}

void set_uniform(NN_Tensor& tensor) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(0.f, 1.f);

	size_t size = tensor.get_elem_size();
	float* tmp = new float[size];

	for (size_t i = 0; i < size; ++i) tmp[i] = dis(gen);
	check_cuda(cudaMemcpy(tensor.data, tmp, sizeof(float) * size, cudaMemcpyHostToDevice));
	delete[] tmp;
}