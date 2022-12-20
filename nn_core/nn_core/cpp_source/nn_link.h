#pragma once
#include "nn_base_layer.h"


template <class _T>
class NN_Coupler {
public:
	vector<NN_Ptr<_T>> object;

	NN_Coupler(const NN_Ptr<_T>& p);
	NN_Coupler(const vector<NN_Ptr<_T>>& p);
	NN_Coupler(const vector<NN_Coupler<_T>>& p);

	NN_Ptr<_T> operator[](int index);

	vector<NN_Ptr<_T>>& get();
};

template <class _T>
NN_Coupler<_T>::NN_Coupler(const NN_Ptr<_T>& p) {
	object.push_back(p);
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(const vector<NN_Ptr<_T>>& p) {
	object = p;
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(const vector<NN_Coupler<_T>>& p) {
	for (NN_Coupler& p_coupler : p) {
		for (NN_Ptr<_T>& p_ptr : p_coupler.object) {
			object.push_back(p_ptr);
		}
	}
}

template <class _T>
NN_Ptr<_T> NN_Coupler<_T>::operator[](int index) {
	return object[index];
}

template <class _T>
vector<NN_Ptr<_T>>& NN_Coupler<_T>::get() {
	return object;
}


class NN_Link {
public:
	bool is_selected;
	bool trainable;

	vector<NN_Ptr<NN_Link>> prev_link;
	vector<NN_Ptr<NN_Link>> next_link;

	NN_Ptr<NN_Link> parent;
	vector<NN_Ptr<NN_Link>> child;

	vector<NN_Ptr<NN_Tensor>> input;
	NN_Ptr<NN_Tensor> output;

	vector<NN_Ptr<NN_Tensor>> d_output;
	NN_Ptr<NN_Tensor> d_input;

	vector<Dim*> input_shape;
	Dim output_shape;

	NN_Ptr<NN_Layer> layer;

	NN_Link(NN_Ptr<NN_Layer> p_layer);
	NN_Link(NN_Ptr<NN_Link> parent_link);

	NN_Coupler<NN_Link> operator()(NN_Coupler<NN_Link>& m_prev_link);
	void operator()(NN_Ptr<NN_Link>& m_prev_link);

	virtual NN_Ptr<NN_Tensor> get_prev_output(NN_Ptr<NN_Link>& p_current);
	virtual NN_Ptr<NN_Tensor> get_next_dinput(NN_Ptr<NN_Link>& p_current);
	virtual Dim get_next_output_shape(NN_Ptr<NN_Link>& p_current);
};