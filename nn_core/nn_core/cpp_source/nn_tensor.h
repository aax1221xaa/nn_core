#pragma once
#include "cuda_common.h"
#include "Dimension.h"



enum { CPU = 1, GPU = 2 };
enum { none = -1, boolean = 1, int8, uint8, int32, uint32, float32, float64 };

class NN_Tensor {
public:
	void* data;
	int attr;

	int dtype;
	NN_Shape shape;
	size_t len;

	NN_Tensor();
	NN_Tensor(const int _attr, const int _dtype);
	~NN_Tensor();

	void clear_tensor();
	void set_tensor(const NN_Shape& _shape, const int _attr, const int _dtype);
	void copy_to(NN_Tensor& dst);

	static size_t get_elem_size(const int type);
};

typedef shared_ptr<NN_Tensor> NN_Tensor_t;