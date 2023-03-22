#pragma once
#include "nn_base_layer.h"


template <class _T>
struct Link_Param {
	_T* link;
	_T* sub_link;
};

template <class _T>
class NN_Coupler : public NN_Shared_Ptr {
public:
	Link_Param<_T>* param;
	int len;

	class Iterator {
	public:
		Link_Param<_T>* p_param;
		int index;

		Iterator(Link_Param<_T>* _param, int _index);
		Iterator(typename const Iterator& p);

		typename const Iterator& operator++();
		typename const Iterator& operator--();
		bool operator!=(typename const Iterator& p) const;
		bool operator==(typename const Iterator& p) const;
		Link_Param<_T>& operator*() const;
	};

	NN_Coupler();
	NN_Coupler(_T* _link);
	NN_Coupler(const Link_Param<_T>& _param);
	NN_Coupler(const vector<Link_Param<_T>>& _params);
	NN_Coupler(const NN_Coupler<_T>& p);
	NN_Coupler(const initializer_list<NN_Coupler<_T>>& list);
	~NN_Coupler();

	void clear();

	const NN_Coupler<_T>& operator+(const NN_Coupler<_T>& p_right);
	const NN_Coupler<_T>& operator=(const NN_Coupler<_T>& p_right);
	Link_Param<_T>& operator[](int index);

	typename const Iterator begin() const;
	typename const Iterator end() const;
};


template <class _T>
NN_Coupler<_T>::Iterator::Iterator(Link_Param<_T>* _param, int _index) {
	p_param = _param;
	index = _index;
}

template <class _T>
NN_Coupler<_T>::Iterator::Iterator(typename const NN_Coupler<_T>::Iterator& p) :
	p_param(p.p_param),
	index(p.index)
{
}

template <class _T>
typename const NN_Coupler<_T>::Iterator& NN_Coupler<_T>::Iterator::operator++() {
	++index;

	return *this;
}

template <class _T>
typename const NN_Coupler<_T>::Iterator& NN_Coupler<_T>::Iterator::operator--() {
	--index;

	return *this;
}

template <class _T>
bool NN_Coupler<_T>::Iterator::operator!=(typename const NN_Coupler<_T>::Iterator& p) const {
	return index != p.index;
}

template <class _T>
bool NN_Coupler<_T>::Iterator::operator==(typename const NN_Coupler<_T>::Iterator& p) const {
	return index == p.index;
}

template <class _T>
Link_Param<_T>& NN_Coupler<_T>::Iterator::operator*() const {
	return p_param[index];
}

template <class _T>
NN_Coupler<_T>::NN_Coupler() {
	param = NULL;
	len = 0;
	id = NULL;
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(_T* _link) {
	param = new Link_Param<_T>;

	param->link = _link;
	param->sub_link = _link;

	len = 1;
	id = linker.Create();
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(const Link_Param<_T>& _param) {
	param = new Link_Param<_T>;

	*param = _param;

	len = 1;
	id = linker.Create();
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(const vector<Link_Param<_T>>& _params) {
	len = (int)_params.size();
	param = new Link_Param<_T>[len];

	for (int i = 0; i < len; ++i) param[i] = _params[i];

	id = linker.Create();
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(const NN_Coupler<_T>& p) {
	id = p.id;
	param = p.param;
	len = p.len;

	if (id) ++id->ref_cnt;
}

template <class _T>
NN_Coupler<_T>::NN_Coupler(const initializer_list<NN_Coupler>& list) :
	NN_Coupler()
{
	for (const NN_Coupler<_T>& p_coupler : list) len += p_coupler.len;
	
	param = new Link_Param<_T>[len];
	int j = 0;
	for (const NN_Coupler<_T>& p_coupler : list) {
		for (int i = 0; i < p_coupler.len; ++i) param[j++] = p_coupler.param[i];
	}

	id = linker.Create();
}

template <class _T>
NN_Coupler<_T>::~NN_Coupler() {
	clear();
}

template <class _T>
void NN_Coupler<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			delete[] param;
			linker.Erase(id);
		}
	}

	param = NULL;
	id = NULL;
	len = 0;
}

template <class _T>
const NN_Coupler<_T>& NN_Coupler<_T>::operator+(const NN_Coupler<_T>& p_right) {
	return NN_Coupler<_T>({ *this, p_right });
}

template <class _T>
const NN_Coupler<_T>& NN_Coupler<_T>::operator=(const NN_Coupler<_T>& p_right) {
	if (&p_right == this) return *this;

	clear();

	id = p_right.id;
	param = p_right.param;
	len = p_right.len;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
Link_Param<_T>& NN_Coupler<_T>::operator[](int index) {
	int m_index = 0;

	if (index < 0) {
		m_index = len + index;
	}
	else {
		m_index = index;
	}

	if (m_index < 0 || m_index >= len) {
		ErrorExcept(
			"[NN_Coupler<_T>::operator[]] index is out of range."
		);
	}

	return param[m_index];
}

template <class _T>
typename const NN_Coupler<_T>::Iterator NN_Coupler<_T>::begin() const {
	return NN_Coupler<_T>::Iterator(param, 0);
}

template <class _T>
typename const NN_Coupler<_T>::Iterator NN_Coupler<_T>::end() const {
	return NN_Coupler<_T>::Iterator(param, len);
}

class NN_Link {
public:
	bool is_selected;
	bool trainable;

	int period;

	vector<NN_Link*> prev_link;
	vector<NN_Link*> next_link;

	NN_Link* parent;
	vector<NN_Link*> child;

	vector<NN_Tensor_t> input;
	NN_Tensor output;

	vector<NN_Tensor_t> d_input;
	vector<NN_Tensor_t> d_output;

	vector<NN_Shape_t> in_shape;
	NN_Shape out_shape;

	NN_Layer_t op_layer;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_Coupler<NN_Link> operator()(NN_Coupler<NN_Link> m_prev_link);
	void inner_link(NN_Link* m_prev_link);

	virtual NN_Link* get_prev_ptr(NN_Link* p_current);
	virtual NN_Link* get_current_ptr(NN_Link* p_prev);

	virtual NN_Link* create_child_link();
};

typedef NN_Coupler<NN_Link> NN;