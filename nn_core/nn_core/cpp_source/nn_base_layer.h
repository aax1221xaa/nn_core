#pragma once
#include "cuda_common.h"
#include "nn_tensor.h"




template <class _T>
class NN_Vec {
public:
	vector<_T> arr;

	NN_Vec(const _T& p);
	NN_Vec(const vector<_T>& p);
	NN_Vec(const vector<NN_Vec<_T>>& p);

	_T operator[](int index) const;

	void clear();
};

template <class _T>
NN_Vec<_T>::NN_Vec(const _T& p) {
	arr.push_back(p);
}

template <class _T>
NN_Vec<_T>::NN_Vec(const vector<_T>& p) {
	arr = p;
}

template <class _T>
NN_Vec<_T>::NN_Vec(const vector<NN_Vec<_T>>& p) {
	for (NN_Vec& p_coupler : p) {
		for (_T& p_ptr : p_coupler.arr) {
			arr.push_back(p_ptr);
		}
	}
}

template <class _T>
_T NN_Vec<_T>::operator[](int index) const {
	return arr[index];
}

template <class _T>
void NN_Vec<_T>::clear() {
	arr.clear();
}

class NN_Layer {
public:
	string name;

	NN_Layer();
	virtual ~NN_Layer();

	virtual void calculate_output_size(vector<NN_Shape_t>& input_shape, NN_Shape_t& output_shape) = 0;
	virtual void build(vector<NN_Shape_t>& input_shape);
	virtual void run_forward(vector<NN_Tensor_t>& input, NN_Tensor_t& output) = 0;
	virtual void run_backward(vector<NN_Tensor_t>& d_output, NN_Tensor_t& d_input) = 0;
};