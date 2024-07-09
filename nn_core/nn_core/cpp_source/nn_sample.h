#pragma once
#include "nn_list.h"
#include "nn_tensor.h"


template <typename _xT, typename _yT>
struct DataSet {
	std::vector<Tensor<_xT>> _x;
	std::vector<Tensor<_yT>> _y;
};


template <typename _xT, typename _yT>
class Sample {
private:
	const DataSet<_xT, _yT>& _origin;
	const int _n_batch;
	const int _n_iter;
	const bool _shuffle;

	static void set_batch_samples(const DataSet<_xT, _yT>& origin, DataSet<_xT, _yT>& buff, int index, int n_batch, bool shuffle);

public:
	class Iterator {
	public:
		const Sample& _p_samples;
		DataSet<_xT, _yT> _buff;
		int _n_iter;

		Iterator(const Sample& p_samples, int n_iter);
		Iterator(const typename Iterator& p);

		bool operator!=(const typename Iterator& p);
		void operator++();
		DataSet<_xT, _yT>& operator*();
	};

	class ConstIterator {
	public:
		const Sample& _p_samples;
		DataSet<_xT, _yT> _buff;
		int _n_iter;

		ConstIterator(const Sample& p_samples, int n_iter);
		ConstIterator(const typename ConstIterator& p);

		bool operator!=(const typename ConstIterator& p) const;
		void operator++();
		const DataSet<_xT, _yT>& operator*() const;
	};

	Sample(const DataSet<_xT, _yT>& current_object, int n_batch, int n_iter, bool shuffle);

	typename Iterator begin();
	typename Iterator end();

	typename ConstIterator begin() const;
	typename ConstIterator end() const;

	DataSet<_xT, _yT> operator[](int index);
	int get_batch();
	int get_iteration();

	const DataSet<_xT, _yT> operator[](int index) const;

	int get_batch() const;
	int get_iteration() const;
};

template <typename _xT, typename _yT>
void Sample<_xT, _yT>::set_batch_samples(const DataSet<_xT, _yT>& origin, DataSet<_xT, _yT>& buff, int index, int n_batch, bool shuffle) {
	std::vector<int> batch_indice;
	typename std::vector<Tensor<_xT>>::iterator x_iter = buff._x.begin();
	typename std::vector<Tensor<_yT>>::iterator y_iter = buff._y.begin();

	for (const NN_List<Tensor<_xT>>& x : origin._x) {
		if (batch_indice.empty()) {
			const int amounts = x.val().get_shape()[0];

			if (shuffle) {
				batch_indice = random_choice(0, amounts, n_batch, shuffle);
			}
			else {
				batch_indice.resize(n_batch);

				for (int i = 0; i < n_batch; ++i) {
					batch_indice[i] = (n_batch * index + i) % amounts;
				}
			}
		}

		*x_iter = x.val()(batch_indice);
		++x_iter;
	}

	for (const NN_List<Tensor<_yT>>& y : origin._y) {
		*y_iter = y.val()(batch_indice);
		++y_iter;
	}
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Iterator::Iterator(const Sample& p_samples, int n_iter) :
	_p_samples(p_samples),
	_n_iter(n_iter)
{
	_buff._x.resize(p_samples._origin._x.size());
	_buff._y.resize(p_samples._origin._y.size());

	set_batch_samples(_p_samples._origin, _buff, _n_iter, _p_samples._n_batch, _p_samples._shuffle);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Iterator::Iterator(const typename Iterator& p) :
	_p_samples(p._p_samples),
	_n_iter(p._n_iter),
	_buff(p._buff)
{
}

template <typename _xT, typename _yT>
bool Sample<_xT, _yT>::Iterator::operator!=(const typename Iterator& p) {
	return _n_iter != p._n_iter;
}

template <typename _xT, typename _yT>
void Sample<_xT, _yT>::Iterator::operator++() {
	set_batch_samples(_p_samples._origin, _buff, _n_iter, _p_samples._n_batch, _p_samples._shuffle);
	++_n_iter;
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>& Sample<_xT, _yT>::Iterator::operator*() {
	return _buff;
}

template <typename _xT, typename _yT>
typename Sample<_xT, _yT>::Iterator Sample<_xT, _yT>::begin() {
	return Iterator(*this, 0);
}

template <typename _xT, typename _yT>
typename Sample<_xT, _yT>::Iterator Sample<_xT, _yT>::end() {
	return Iterator(*this, _n_iter);
}

template <typename _xT, typename _yT>
typename Sample<_xT, _yT>::ConstIterator Sample<_xT, _yT>::begin() const {
	return ConstIterator(*this, 0);
}

template <typename _xT, typename _yT>
typename Sample<_xT, _yT>::ConstIterator Sample<_xT, _yT>::end() const {
	return ConstIterator(*this, _n_iter);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::ConstIterator::ConstIterator(const Sample& p_samples, int n_iter) :
	_p_samples(p_samples),
	_n_iter(n_iter)
{
	_buff._x.resize(p_samples._origin._x.size());
	_buff._y.resize(p_samples._origin._y.size());

	set_batch_samples(_p_samples._origin, _buff, _n_iter, _p_samples._n_batch, _p_samples._shuffle);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::ConstIterator::ConstIterator(const typename ConstIterator& p) :
	_p_samples(p._p_samples),
	_n_iter(p._n_iter),
	_buff(p._buff)
{
}

template <typename _xT, typename _yT>
bool Sample<_xT, _yT>::ConstIterator::operator!=(const typename ConstIterator& p) const {
	return _n_iter != p._n_iter;
}

template <typename _xT, typename _yT>
void Sample<_xT, _yT>::ConstIterator::operator++() {
	set_batch_samples(_p_samples._origin, _buff, _n_iter, _p_samples._n_batch, _p_samples._shuffle);
	++_n_iter;
}

template <typename _xT, typename _yT>
const DataSet<_xT, _yT>& Sample<_xT, _yT>::ConstIterator::operator*() const {
	return _buff;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Sample(const DataSet<_xT, _yT>& current_object, int n_batch, int n_iter, bool shuffle) :
	_origin(current_object),
	_n_batch(n_batch),
	_n_iter(n_iter),
	_shuffle(shuffle)
{
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT> Sample<_xT, _yT>::operator[](int index) {
	if (index >= _n_iter) {
		ErrorExcept(
			"[MNIST::Sample::operator[]] Index[%d] is out of range. (%d ~ %d)",
			index,
			0, _n_iter
		);
	}

	DataSet<_xT, _yT> buff;

	buff._x.resize(_origin._x.size());
	buff._y.resize(_origin._y.size());

	set_batch_samples(_origin, buff, index, _n_batch, _shuffle);

	return buff;
}

template <typename _xT, typename _yT>
int Sample<_xT, _yT>::get_batch() {
	return _n_batch;
}

template <typename _xT, typename _yT>
int Sample<_xT, _yT>::get_iteration() {
	return _n_iter;
}

template <typename _xT, typename _yT>
const DataSet<_xT, _yT> Sample<_xT, _yT>::operator[](int index) const {
	if (index >= _n_iter) {
		ErrorExcept(
			"[MNIST::Sample::operator[]] Index[%d] is out of range. (%d ~ %d)",
			index,
			0, _n_iter
		);
	}

	DataSet<_xT, _yT> buff;

	buff._x.resize(_origin._x.size());
	buff._y.resize(_origin._y.size());

	set_batch_samples(_origin, buff, index, _n_batch, _shuffle);

	return buff;
}

template <typename _xT, typename _yT>
int Sample<_xT, _yT>::get_batch() const {
	return _n_batch;
}

template <typename _xT, typename _yT>
int Sample<_xT, _yT>::get_iteration() const {
	return _n_iter;
}