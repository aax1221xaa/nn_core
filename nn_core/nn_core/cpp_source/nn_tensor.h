#pragma once
#include "cuda_common.h"
#include "Dimension.h"



enum { CPU = 1, GPU = 2 };


class NN_Tensor : public NN_Shared_Ptr {
public:
	int device_type;
	float* data;
	size_t bytes;

	int test_value;
	
	NN_Tensor();
	NN_Tensor(const std::vector<int>& _shape, int _device_type);
	NN_Tensor(const size_t _bytes, int _device_type);
	NN_Tensor(const NN_Tensor& p);
	~NN_Tensor();

	const NN_Tensor& operator=(const NN_Tensor& p);

	void clear();
	void set(const std::vector<int>& _shape, int _device_type);
	void copy_to(NN_Tensor& dst, const int _device_type);
	
	static NN_Tensor zeros(const std::vector<int>& _shape, int _device_type);
	static NN_Tensor zeros_like(const NN_Tensor& p, int _device_type);
};

void set_uniform(NN_Tensor& p);