#include "nn_tensor.h"


extern char str_buffer[STR_MAX];
extern int str_idx;


void NN_Tensor::destroy() {
	if (device_type == CPU) {
		free(data);
	}
	else if (device_type == GPU) {
		check_cuda(cudaFree(data));
	}
}

void NN_Tensor::clear_param() {
	id = NULL;
	data = NULL;
	dtype = none;
	size = 0;
}

NN_Tensor::NN_Tensor(int _device_type) :
	device_type(_device_type)
{
}

NN_Tensor::NN_Tensor(const Dim& _shape, int _dtype, int _device_type) :
	device_type(_device_type)
{
	shape = _shape;
	dtype = _dtype;

	alloc(*this, shape, dtype);
	create();
}

NN_Tensor::NN_Tensor(const NN_Tensor& p) :
	device_type(p.device_type)
{
	clear();
	copy_id(p);

	data = p.data;
	shape = p.shape;
	dtype = p.dtype;
	size = p.size;
}

NN_Tensor& NN_Tensor::operator=(const NN_Tensor& p) {
	if (this == &p) return *this;

	if (size != p.size) {
		clear();

		shape = p.shape;
		dtype = p.dtype;
		size = p.size;
	}

	if (device_type == p.device_type) {
		data = p.data;
		copy_id(p);
	}
	else {
		alloc(*this, shape, dtype);

		if (device_type == CPU) {
			check_cuda(cudaMemcpy(data, p.data, size, cudaMemcpyDeviceToHost));
		}
		else {
			check_cuda(cudaMemcpy(data, p.data, size, cudaMemcpyHostToDevice));
		}
		create();
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

void NN_Tensor::alloc(NN_Tensor& tensor, const Dim& shape, int dtype){
	tensor.size = get_type_size(dtype);
	tensor.shape = shape;
	tensor.dtype = dtype;
	
	for (int n : shape.dim) {
		if (n < 0) {
			ErrorExcept(
				"[nn_tensor_host_malloc] invalid size. %s",
				shape_to_str(tensor)
			);
		}
		else {
			tensor.size *= n;
		}
	}

	if (tensor.device_type == CPU) {
		tensor.data = malloc(tensor.size);
	}
	else if (tensor.device_type == GPU) {
		check_cuda(cudaMalloc(&tensor.data, tensor.size));
	}
}

void NN_Tensor::copyto(NN_Tensor& dst) {
	if (src.size != dst.size) {
		ErrorExcept("[nn_tensor_copy] src와 dst 사이즈가 맞지 않습니다. %d != %d", src.size, dst.size);
	}

	if (src.device_type == CPU) {
		if (dst.device_type == CPU) {
			memcpy_s(dst.data, dst.size, src.data, src.size);
		}
		else if (dst.device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, src.data, src.size, cudaMemcpyHostToDevice));
		}
	}
	else if (src.device_type == GPU) {
		if (dst.device_type == CPU) {
			check_cuda(cudaMemcpy(dst.data, src.data, src.size, cudaMemcpyDeviceToHost));
		}
		else if (dst.device_type == GPU) {
			check_cuda(cudaMemcpy(dst.data, src.data, src.size, cudaMemcpyDeviceToDevice));
		}
	}
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