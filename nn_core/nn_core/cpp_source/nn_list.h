#pragma once
#include <memory>
#include "cuda_common.h"
#include "Exception.h"

#define PUT_LIST


/**********************************************/
/*                                            */
/*                   NN_List                  */
/*                                            */
/**********************************************/

template <class _T>
class NN_List {

	_T* _data;
	std::vector<NN_List*>* _p_list;

	NN_List(_T* data);

public:
	class Iterator {
	public:
		bool _sw;
		NN_List& _ref;
		typename std::vector<NN_List*>::iterator _iter;

		Iterator(NN_List& list, typename std::vector<NN_List*>::iterator iter);
		Iterator(const typename NN_List::Iterator& p);

		void operator++();
		bool operator!=(const typename NN_List::Iterator& p);
		NN_List& operator*();
	};

	class ConstIterator {
	public:
		bool _sw;
		const NN_List& _ref;
		typename std::vector<NN_List*>::const_iterator _iter;

		ConstIterator(const NN_List& list, typename std::vector<NN_List*>::const_iterator iter);
		ConstIterator(const typename NN_List::ConstIterator& p);

		void operator++();
		bool operator!=(const typename NN_List::ConstIterator& p) const;
		const NN_List& operator*() const;
	};

	NN_List();
	NN_List(const _T& val);
	NN_List(_T&& val);
	NN_List(const std::initializer_list<_T>& list);
	NN_List(const std::initializer_list<NN_List>& list);
	NN_List(const NN_List& p);
	NN_List(NN_List&& p);
	~NN_List();

	const _T& operator=(const _T& val);
	NN_List& operator=(const NN_List& p);
	NN_List& operator=(NN_List&& p);
	NN_List& operator=(const std::initializer_list<_T>& list);
	NN_List& operator[](int index);

	void clear();
	void append(const _T& val);
	void append(_T&& val);
	void append(const std::initializer_list<_T>& list);
	void append(const NN_List& list);
	void extend(const _T& val);
	void extend(_T&& val);
	void extend(const std::initializer_list<_T>& list);
	void extend(const NN_List& list);
	_T& val();
	size_t size();
	size_t capacity();
	void reserve(size_t size);
	void resize(size_t size);
	void resize(size_t size, const _T& val);
	void resize(size_t size, _T&& val);
	bool is_scalar();

	bool is_scalar() const;
	size_t size() const;
	size_t capacity() const;
	const _T& val() const;

	bool is_empty();
	bool is_empty() const;

	const NN_List& operator[](int index) const;

	typename NN_List::Iterator begin();
	typename NN_List::Iterator end();

	typename NN_List::ConstIterator begin() const;
	typename NN_List::ConstIterator end() const;

#ifdef PUT_LIST
	std::ostream& put_list(std::ostream& os) const;
#endif
};

template <class _T>
NN_List<_T>::Iterator::Iterator(NN_List& list, typename std::vector<NN_List<_T>*>::iterator iter) :
	_ref(list),
	_iter(iter)
{
	if (list._data) _sw = true;
	else _sw = false;
}

template <class _T>
NN_List<_T>::Iterator::Iterator(const typename NN_List::Iterator& p) :
	_ref(p._ref),
	_iter(p._iter),
	_sw(p._sw)
{
}

template <class _T>
void NN_List<_T>::Iterator::operator++() {
	++_iter;
	_sw = false;
}

template <class _T>
bool NN_List<_T>::Iterator::operator!=(const typename NN_List::Iterator& p) {
	if (_sw) return true;
	else return _iter != p._iter;
}

template <class _T>
NN_List<_T>& NN_List<_T>::Iterator::operator*() {
	if (_sw) return _ref;
	else return *(*_iter);
}

template <class _T>
NN_List<_T>::ConstIterator::ConstIterator(const NN_List& list, typename std::vector<NN_List<_T>*>::const_iterator iter) :
	_ref(list),
	_iter(iter)
{
	if (list._data) _sw = true;
	else _sw = false;
}

template <class _T>
NN_List<_T>::ConstIterator::ConstIterator(const typename NN_List::ConstIterator& p) :
	_ref(p._ref),
	_iter(p._iter),
	_sw(p._sw)
{
}

template <class _T>
void NN_List<_T>::ConstIterator::operator++() {
	++_iter;
	_sw = false;
}

template <class _T>
bool NN_List<_T>::ConstIterator::operator!=(const typename NN_List::ConstIterator& p) const {
	if (_sw) return true;
	else return _iter != p._iter;
}

template <class _T>
const NN_List<_T>& NN_List<_T>::ConstIterator::operator*() const {
	if (_sw) return _ref;
	else return *(*_iter);
}

template <class _T>
NN_List<_T>::NN_List(_T* data) :
	_data(data),
	_p_list(NULL)
{
}

template <class _T>
NN_List<_T>::NN_List() :
	_data(NULL),
	_p_list(NULL)
{
}

template <class _T>
NN_List<_T>::NN_List(const _T& val) :
	_data(new _T(val)),
	_p_list(NULL)
{
}

template <class _T>
NN_List<_T>::NN_List(_T&& val) :
	_data(new _T(val)),
	_p_list(NULL)
{
}

template <class _T>
NN_List<_T>::NN_List(const std::initializer_list<_T>& list) :
	_data(NULL),
	_p_list(new std::vector<NN_List*>())
{
	if (_p_list->capacity() < list.size()) _p_list->reserve(list.size());

	for (const _T& elem : list) _p_list->push_back(new NN_List(elem));
}

template <class _T>
NN_List<_T>::NN_List(const std::initializer_list<NN_List>& list) :
	_data(NULL),
	_p_list(new std::vector<NN_List*>())
{
	if (_p_list->capacity() < list.size()) _p_list->reserve(list.size());

	for (const NN_List& m_list : list) _p_list->push_back(new NN_List(m_list));
}

template <class _T>
NN_List<_T>::NN_List(const NN_List& p) :
	_p_list(NULL),
	_data(NULL)
{
	if (p._data) {
		_data = new _T(*p._data);
	}
	else if (p._p_list) {
		_p_list = new std::vector<NN_List*>();

		if (_p_list->capacity() < p._p_list->size()) _p_list->reserve(p._p_list->size());

		for (const NN_List* p_list : *p._p_list) _p_list->push_back(new NN_List(*p_list));
	}
}

template <class _T>
NN_List<_T>::NN_List(NN_List&& p) :
	_data(p._data),
	_p_list(p._p_list)
{
	p._data = NULL;
	p._p_list = NULL;
}

template <class _T>
NN_List<_T>::~NN_List() {
	delete _data;

	if (_p_list) {
		for (NN_List* p_list : *_p_list) delete p_list;
	}

	delete _p_list;
}

template <class _T>
const _T& NN_List<_T>::operator=(const _T& val) {
	clear();

	_data = new _T(val);

	return val;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator=(const NN_List& p) {
	if (&p == this) return *this;

	clear();

	if (p._data) {
		_data = new _T(*p._data);
	}
	else if (p._p_list) {
		_p_list = new std::vector<NN_List*>();

		if (_p_list->capacity() < p._p_list->size()) _p_list->reserve(p._p_list->size());

		for (const NN_List* p_list : *p._p_list) _p_list->push_back(new NN_List(*p_list));
	}

	return *this;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator=(NN_List&& p) {
	if (&p == this) return *this;

	clear();

	_data = p._data;
	_p_list = p._p_list;

	p._data = NULL;
	p._p_list = NULL;

	return *this;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator=(const std::initializer_list<_T>& list) {
	clear();

	_p_list = new std::vector<NN_List*>();

	if (_p_list->capacity() < list.size()) _p_list->reserve(list.size());

	for (const _T& p_elem : list) _p_list->push_back(new NN_List(p_elem));

	return *this;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator[](int index) {
	if (_data) {
		if (index != 0) {
			ErrorExcept(
				"[NN_List<_T>::operator[]] Scalar can't index."
			);
		}

		return *this;
	}
	else if (_p_list) {
		const int len = (int)_p_list->size();

		index = index < 0 ? len + index : index;

		if (index >= len || index < 0) {
			ErrorExcept(
				"[NN_List<_T>::operator[]] Index is out of range. (0 ~ %d)",
				len - 1
			);
		}
	}
	else {
		ErrorExcept(
			"[NN_List<_T>::operator[]] This list is none."
		);
	}

	return *(*_p_list)[index];
}

template <class _T>
void NN_List<_T>::clear() {
	delete _data;

	if (_p_list) {
		for (NN_List* p_list : *_p_list) delete p_list;
	}

	delete _p_list;

	_data = NULL;
	_p_list = NULL;
}

template <class _T>
void NN_List<_T>::append(const _T& val) {
	if (!_p_list) _p_list = new std::vector<NN_List*>();

	if (_data) {
		_p_list->push_back(new NN_List(_data));
		_data = NULL;
	}
	
	_p_list->push_back(new NN_List(val));
}

template <class _T>
void NN_List<_T>::append(_T&& val) {
	if (!_p_list) _p_list = new std::vector<NN_List*>();

	if (_data) {
		_p_list->push_back(new NN_List(_data));
		_data = NULL;
	}

	_p_list->push_back(new NN_List(val));
}

template <class _T>
void NN_List<_T>::append(const std::initializer_list<_T>& list) {
	if (!_p_list) _p_list = new std::vector<NN_List*>();

	if (_data) {
		_p_list->push_back(new NN_List(_data));
		_data = NULL;
	}

	_p_list->push_back(new NN_List(list));
}

template <class _T>
void NN_List<_T>::append(const NN_List& list) {
	if (!_p_list) _p_list = new std::vector<NN_List*>();

	if (_data) {
		_p_list->push_back(new NN_List(_data));
		_data = NULL;
	}

	_p_list->push_back(new NN_List(list));
}

template <class _T>
void NN_List<_T>::extend(const _T& val) {
	if (!_p_list) _p_list = new std::vector<NN_List*>();

	if (_data) {
		_p_list->push_back(new NN_List(_data));
		_data = NULL;
	}

	_p_list->push_back(new NN_List(val));
}

template <class _T>
void NN_List<_T>::extend(_T&& val) {
	if (!_p_list) _p_list = new std::vector<NN_List*>();

	if (_data) {
		_p_list->push_back(new NN_List(_data));
		_data = NULL;
	}

	_p_list->push_back(new NN_List(val));
}

template <class _T>
void NN_List<_T>::extend(const std::initializer_list<_T>& list) {
	if (!_p_list) {
		_p_list = new std::vector<NN_List*>();

		const size_t src_size = list.size();
		const size_t dst_cap = _p_list->capacity();

		if (_data) {
			if (dst_cap < src_size + 1) _p_list->reserve(src_size + 1);

			_p_list->push_back(new NN_List(_data));
			_data = NULL;
		}
		else {
			if (dst_cap < src_size) _p_list->reserve(src_size);
		}
	}

	for (const _T& p_elem : list) _p_list->push_back(new NN_List(p_elem));
}

template <class _T>
void NN_List<_T>::extend(const NN_List& list) {
	if (!_p_list) {
		_p_list = new std::vector<NN_List*>();

		const size_t src_size = list.size();
		const size_t dst_cap = _p_list->capacity();

		if (_data) {
			if (dst_cap < src_size + 1) _p_list->reserve(src_size + 1);

			_p_list->push_back(new NN_List(_data));
			_data = NULL;
		}
		else {
			if (dst_cap < src_size) _p_list->reserve(src_size);
		}
	}

	for (const NN_List& m_list : list) _p_list->push_back(new NN_List(m_list));
}

template <class _T>
_T& NN_List<_T>::val() {
	if (!_data) {
		if (!_p_list) {
			ErrorExcept(
				"[NN_List::val] This list is none."
			);
		}
		else {
			if (_p_list->size() > 1) {
				ErrorExcept(
					"[NN_List::val] This list's elements amounts not one. size = %ld",
					_p_list->size()
				);
			}
		}

		return *(*_p_list).front()->_data;
	}

	return *_data;
}

template <class _T>
size_t NN_List<_T>::size() {
	if (_data) return 1;
	else if (_p_list) return _p_list->size();
	else return 0;
}

template <class _T>
size_t NN_List<_T>::capacity() {
	if (_p_list) return _p_list->capacity();
	else return 0;
}

template <class _T>
void NN_List<_T>::reserve(size_t size) {
	clear();
	
	_p_list = new std::vector<NN_List*>();
	_p_list->reserve(size);
}

template <class _T>
void NN_List<_T>::resize(size_t size) {
	clear();

	_p_list = new std::vector<NN_List*>(size, NULL);

	for (NN_List*& p_list : *_p_list) p_list = new NN_List();
}

template <class _T>
void NN_List<_T>::resize(size_t size, const _T& val) {
	clear();

	_p_list = new std::vector<NN_List*>(size, NULL);

	for (NN_List*& p_list : *_p_list) p_list = new NN_List(val);
}

template <class _T>
void NN_List<_T>::resize(size_t size, _T&& val) {
	clear();

	_p_list = new std::vector<NN_List*>(size, NULL);

	for (NN_List*& p_list : *_p_list) p_list = new NN_List(val);
}

template <class _T>
bool NN_List<_T>::is_scalar() {
	if (_data) return true;
	else return false;
}

template <class _T>
bool NN_List<_T>::is_scalar() const {
	if (_data) return true;
	else return false;
}

template <class _T>
bool NN_List<_T>::is_empty() {
	if (_data == NULL && _p_list == NULL) return true;
	else return false;
}

template <class _T>
bool NN_List<_T>::is_empty() const {
	if (_data == NULL && _p_list == NULL) return true;
	else return false;
}

template <class _T>
size_t NN_List<_T>::size() const {
	if (_data) return 1;
	else if (_p_list) return _p_list->size();
	else return 0;
}

template <class _T>
size_t NN_List<_T>::capacity() const {
	if (_p_list) return _p_list->capacity();
	else return 0;
}

template <class _T>
const _T& NN_List<_T>::val() const {
	if (!_data) {
		if (!_p_list) {
			ErrorExcept(
				"[NN_List::val] This list is none."
			);
		}
		else {
			if (_p_list->size() > 1) {
				ErrorExcept(
					"[NN_List::val] This list's elements amounts not one. size = %ld",
					_p_list->size()
				);
			}
		}

		return *(*_p_list).front()->_data;
	}

	return *_data;
}

template <class _T>
const NN_List<_T>& NN_List<_T>::operator[](int index) const {
	if (_data) {
		if (index != 0) {
			ErrorExcept(
				"[NN_List<_T>::operator[]] Scalar can't index."
			);
		}

		return *this;
	}
	else if (_p_list) {
		const int len = (int)_p_list->size();

		index = index < 0 ? len + index : index;

		if (index >= len || index < 0) {
			ErrorExcept(
				"[NN_List<_T>::operator[]] Index is out of range. (0 ~ %d)",
				len - 1
			);
		}
	}
	else {
		ErrorExcept(
			"[NN_List<_T>::operator[]] This list is none."
		);
	}

	return *(*_p_list)[index];
}

template <class _T>
typename NN_List<_T>::Iterator NN_List<_T>::begin() {
	return NN_List<_T>::Iterator(*this, _p_list->begin());
}

template <class _T>
typename NN_List<_T>::Iterator NN_List<_T>::end() {
	return NN_List<_T>::Iterator(*this, _p_list->end());
}

template <class _T>
typename NN_List<_T>::ConstIterator NN_List<_T>::begin() const {
	return NN_List<_T>::ConstIterator(*this, _p_list->begin());
}

template <class _T>
typename NN_List<_T>::ConstIterator NN_List<_T>::end() const {
	return NN_List<_T>::ConstIterator(*this, _p_list->end());
}

#ifdef PUT_LIST
template <class _T>
std::ostream& NN_List<_T>::put_list(std::ostream& os) const {
	if (_data) os << *_data;
	else {
		os << '[';

		if (_p_list) {
			for (const NN_List* p_list : *_p_list) {
				if (p_list) {
					p_list->put_list(os);
				}

				os << ", ";
			}
		}

		os << ']';
	}

	return os;
}

template <class _T>
std::ostream& operator<<(std::ostream& os, const NN_List<_T>& list) {
	return list.put_list(os) << std::endl;
}
#endif
