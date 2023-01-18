#pragma once
#include "cuda_common.h"
#include "Dimension.h"



enum { CPU = 1, GPU = 2 };
//enum { none = -1, boolean = 1, int8, uint8, int32, uint32, float32, float64 };


class NN_Tensor {
public:
	float* data;
	NN_Shape shape;

	const int device_type;

	NN_Tensor(const int _device_type);
	NN_Tensor(const NN_Shape& _shape, const int _device_type);
	~NN_Tensor();

	void clear();
	void create(const NN_Shape& _shape);
	void copy_to(NN_Tensor& dst);

	const size_t get_elem_size() const;
};

typedef shared_ptr<NN_Tensor> NN_Tensor_t;