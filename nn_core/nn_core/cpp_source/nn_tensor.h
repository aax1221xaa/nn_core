#pragma once
#include "cuda_common.h"
#include "nn_manager.h"
#include "Dimension.h"


enum { none = -1, boolean = 1, int8, uint8, int32, uint32, float32, float64 };

class NN_Tensor : public NN_Manager {
protected:
	void destroy();
	void clear_param();

public:
	void* data;

	Dim shape;
	int dtype;
	const int device_type;

	size_t size;

	NN_Tensor(int _device_type);
	NN_Tensor(const Dim& _shape, int _dtype, int _device_type);
	NN_Tensor(const NN_Tensor& p);

	NN_Tensor& operator=(const NN_Tensor& p);

	void copyto(NN_Tensor& dst);
	
	static const size_t get_type_size(int type);
	static const char* shape_to_str(const NN_Tensor& tensor);
	static void alloc(NN_Tensor& tensor, const Dim& shape, int dtype);
};