#pragma once
#include "nn_base_layer.h"


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

template <class _T>
struct NN_Coupler {
	_T* link;
	NN_Tensor* output;
	NN_Tensor* d_input;
	NN_Shape* out_size;
};

class NN_Link {
public:
	bool is_selected;
	bool trainable;

	vector<NN_Link*> prev_link;
	vector<NN_Link*> next_link;

	NN_Link* parent;
	vector<NN_Link*> child;

	vector<NN_Tensor*> input;
	NN_Tensor output;

	NN_Tensor d_input;
	vector<NN_Tensor*> d_output;

	vector<NN_Shape*> in_shape;
	NN_Shape out_shape;

	NN_Layer* op_layer;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Vec<NN_Coupler<NN_Link>> operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link);
	void operator()(NN_Link* m_prev_link);

	virtual NN_Link* create_child_link();
};

typedef NN_Vec<NN_Coupler<NN_Link>> NN;