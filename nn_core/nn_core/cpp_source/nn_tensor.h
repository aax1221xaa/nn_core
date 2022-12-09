#pragma once
#include "cuda_common.h"
#include "Dimension.h"


enum { none = -1, boolean = 1, int8, uint8, int32, uint32, float32, float64 };

class NN_Tensor {
public:
	void* data;

	Dim shape;
	int dtype;
	int device_type;

	size_t size;

	NN_Tensor();
	NN_Tensor(const Dim& _shape, int _dtype, int _device_type);
	~NN_Tensor();

	void copyto(NN_Tensor& dst);
	
	static const size_t get_type_size(int type);
	static const char* shape_to_str(const NN_Tensor& tensor);
};