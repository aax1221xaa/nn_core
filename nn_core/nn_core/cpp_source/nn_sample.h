#pragma once
#include "nn_list.h"
#include "nn_tensor.h"


template <typename _xT, typename _yT>
struct Sample {
	NN_List<Tensor<_xT>> _x;
	NN_List<Tensor<_yT>> _y;
};

template <typename _xT, typename _yT>
class DataSet {
	virtual Sample<_xT, _yT> get_samples(int index) const;
	virtual int get_end_index() const;

public:
	class Iterator {
	public:
		const DataSet& _dataset;
		int _index;

		Iterator(const DataSet& dataset, int index);
		Iterator(const typename Iterator& p);

		bool operator!=(const typename Iterator& p);
		void operator++();
		Sample<_xT, _yT> operator*();
	};

	class ConstIterator {
	public:
		const DataSet& _dataset;
		int _index;

		ConstIterator(const DataSet& dataset, int index);
		ConstIterator(const typename ConstIterator& p);

		bool operator!=(const typename ConstIterator& p) const;
		void operator++();
		Sample<_xT, _yT> operator*() const;
	};

	DataSet();
	virtual ~DataSet();

	typename Iterator begin();
	typename Iterator end();

	typename ConstIterator begin() const;
	typename ConstIterator end() const;

	Sample<_xT, _yT> operator[](int index);
	Sample<_xT, _yT> operator[](int index) const;
};

template <typename _xT, typename _yT>
Sample<_xT, _yT> DataSet<_xT, _yT>::get_samples(int index) const {
	return Sample<_xT, _yT>();
}

template <typename _xT, typename _yT>
int DataSet<_xT, _yT>::get_end_index() const {
	return 0;
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>::Iterator::Iterator(const DataSet& dataset, int index) :
	_dataset(dataset),
	_index(index)
{

}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>::Iterator::Iterator(const typename DataSet::Iterator& p) :
	_dataset(p._dataset),
	_index(p._index)
{
}

template <typename _xT, typename _yT>
bool DataSet<_xT, _yT>::Iterator::operator!=(const typename DataSet::Iterator& p) {
	return _index != p._index;
}

template <typename _xT, typename _yT>
void DataSet<_xT, _yT>::Iterator::operator++() {
	++_index;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> DataSet<_xT, _yT>::Iterator::operator*() {
	return _dataset.get_samples(_index);
}

template <typename _xT, typename _yT>
typename DataSet<_xT, _yT>::Iterator DataSet<_xT, _yT>::begin() {
	return Iterator(*this, 0);
}

template <typename _xT, typename _yT>
typename DataSet<_xT, _yT>::Iterator DataSet<_xT, _yT>::end() {
	return Iterator(*this, get_end_index());
}

template <typename _xT, typename _yT>
typename DataSet<_xT, _yT>::ConstIterator DataSet<_xT, _yT>::begin() const {
	return ConstIterator(*this, 0);
}

template <typename _xT, typename _yT>
typename DataSet<_xT, _yT>::ConstIterator DataSet<_xT, _yT>::end() const {
	return ConstIterator(*this, get_end_index());
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>::ConstIterator::ConstIterator(const DataSet& dataset, int index) :
	_dataset(dataset),
	_index(index)
{

}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>::ConstIterator::ConstIterator(const typename DataSet::ConstIterator& p) :
	_dataset(p._dataset),
	_index(p._index)
{
}

template <typename _xT, typename _yT>
bool DataSet<_xT, _yT>::ConstIterator::operator!=(const typename DataSet::ConstIterator& p) const {
	return _index != p._index;
}

template <typename _xT, typename _yT>
void DataSet<_xT, _yT>::ConstIterator::operator++() {
	++_index;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> DataSet<_xT, _yT>::ConstIterator::operator*() const {
	return _dataset.get_samples(_index);
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>::DataSet() {

}

template <typename _xT, typename _yT>
DataSet<_xT, _yT>::~DataSet() {

}

template <typename _xT, typename _yT>
Sample<_xT, _yT> DataSet<_xT, _yT>::operator[](int index) {
	return get_samples(index);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> DataSet<_xT, _yT>::operator[](int index) const {
	return get_samples(index);
}