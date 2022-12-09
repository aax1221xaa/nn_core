#include "nn_tensor.h"


extern char str_buffer[STR_MAX];
extern int str_idx;


NN_Tensor::NN_Tensor() {
	data = NULL;
	dtype = none;
	device_type = none;
	size = 0;
}

NN_Tensor::NN_Tensor(const Dim& _shape, int _dtype, int _device_type) {
	shape = _shape;
	dtype = _dtype;
	device_type = _device_type;

	size = get_type_size(dtype);

	try {
		for (int n : shape.dim) {
			if (n < 0) {
				ErrorExcept(
					"[nn_tensor_host_malloc] invalid size. %s",
					shape_to_str(*this)
				);
			}
			else {
				size *= n;
			}
		}

		if (device_type == CPU) {
			data = malloc(size);
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
		if (device_type == CPU) {
			free(data);
		}
		else if (device_type == GPU) {
			check_cuda(cudaFree(data));
		}
	}
	catch (Exception& e) {
		e.Put();
	}
}

void NN_Tensor::copyto(NN_Tensor& dst) {
	if (size != dst.size) {
		ErrorExcept("[nn_tensor_copy] src와 dst 사이즈가 맞지 않습니다. %d != %d", size, dst.size);
	}

	if (device_type == CPU) {
		if (dst.device_type == CPU) {
			memcpy_s(dst.data, dst.size, data, size);
		}
		else if (dst.device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, size, cudaMemcpyHostToDevice));
		}
	}
	else if (device_type == GPU) {
		if (dst.device_type == CPU) {
			check_cuda(cudaMemcpy(dst.data, data, size, cudaMemcpyDeviceToHost));
		}
		else if (dst.device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, data, size, cudaMemcpyDeviceToDevice));
		}
	}
}

const size_t NN_Tensor::get_type_size(int type) {
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

const char* NN_Tensor::shape_to_str(const NN_Tensor& tensor) {
	char tmp_buff[128] = { '\0', };
	char elem[32] = { '\0', };

	sprintf_s(tmp_buff, "[");
	for (int n : tensor.shape.dim) {
		sprintf_s(elem, "%d, ", n);
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