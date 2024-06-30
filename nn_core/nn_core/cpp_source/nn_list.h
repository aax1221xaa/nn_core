#pragma once
#include "cuda_common.h"




/**********************************************/
/*                                            */
/*                   NN_List                  */
/*                                            */
/**********************************************/

template <class _T>
class NN_List {
	std::shared_ptr<_T> _data;
	std::vector<NN_List*> _p_list;

	NN_List(std::shared_ptr<_T>& data);

public:
	class Iterator {
	public:
		typename std::vector<NN_List*>::iterator _iter;

		Iterator(const typename std::vector<NN_List*>::iterator& iter);
		Iterator(const typename Iterator& p);

		void operator++();
		bool operator!=(const typename Iterator& p);
		NN_List& operator*();
	};

	class ConstIterator {
	public:
		typename std::vector<NN_List*>::const_iterator _iter;

		ConstIterator(const typename std::vector<NN_List*>::const_iterator& iter);
		ConstIterator(const typename ConstIterator& p);

		void operator++();
		bool operator!=(const typename ConstIterator& p) const;
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
	void reserve(size_t size);
	void resize(size_t size);

	size_t size() const;
	const _T& val() const;

	const NN_List& operator[](int index) const;

	typename Iterator begin();
	typename Iterator end();

	typename ConstIterator begin() const;
	typename ConstIterator end() const;

#ifdef PUT_LIST
	std::ostream& put_list(std::ostream& os) const;
#endif
};

template <class _T>
NN_List<_T>::Iterator::Iterator(const typename std::vector<NN_List<_T>*>::iterator& iter) :
	_iter(iter)
{
}

template <class _T>
NN_List<_T>::Iterator::Iterator(const typename Iterator& p) :
	_iter(p._iter)
{
}

template <class _T>
void NN_List<_T>::Iterator::operator++() {
	++_iter;
}

template <class _T>
bool NN_List<_T>::Iterator::operator!=(const typename Iterator& p) {
	return _iter != p._iter;
}

template <class _T>
NN_List<_T>& NN_List<_T>::Iterator::operator*() {
	return *(*_iter);
}

template <class _T>
NN_List<_T>::ConstIterator::ConstIterator(const typename std::vector<NN_List<_T>*>::const_iterator& iter) :
	_iter(iter)
{
}

template <class _T>
NN_List<_T>::ConstIterator::ConstIterator(const typename ConstIterator& p) :
	_iter(p._iter)
{
}

template <class _T>
void NN_List<_T>::ConstIterator:: operator++() {
	++_iter;
}

template <class _T>
bool NN_List<_T>::ConstIterator::operator!=(const typename ConstIterator& p) const {
	return _iter != p._iter;
}

template <class _T>
const NN_List<_T>& NN_List<_T>::ConstIterator::operator*() const {
	return *(*_iter);
}

template <class _T>
NN_List<_T>::NN_List(std::shared_ptr<_T>& data) :
	_data(data)
{
	_p_list.push_back(this);
}

template <class _T>
NN_List<_T>::NN_List() :
	_data(NULL)
{
	_p_list.push_back(this);
}

template <class _T>
NN_List<_T>::NN_List(const _T& val) :
	_data(std::shared_ptr<_T>(new _T(val)))
{
	_p_list.push_back(this);
}

template <class _T>
NN_List<_T>::NN_List(_T&& val) :
	_data(std::shared_ptr<_T>(new _T(val)))
{
	_p_list.push_back(this);
}

template <class _T>
NN_List<_T>::NN_List(const std::initializer_list<_T>& list) :
	_data(NULL),
	_p_list(list.size(), NULL)
{
	int i = 0;
	for (const _T& elem : list) _p_list[i++] = new NN_List(elem);
}

template <class _T>
NN_List<_T>::NN_List(const std::initializer_list<NN_List>& list) :
	_data(NULL),
	_p_list(list.size(), NULL)
{
	int i = 0;
	for (const NN_List& m_list : list) _p_list[i++] = new NN_List(m_list);
}

template <class _T>
NN_List<_T>::NN_List(const NN_List& p) :
	_p_list(p._p_list.size(), NULL)
{
	if (p._p_list.front() != &p) {
		int i = 0;

		for (const NN_List* list : p._p_list) _p_list[i++] = new NN_List(*list);
	}
	else {
		_data = p._data;
		_p_list.front() = this;
	}
}

template <class _T>
NN_List<_T>::NN_List(NN_List&& p) :
	_data(std::move(p._data)),
	_p_list(std::move(p._p_list))
{
	if (_p_list.front() == &p) _p_list.front() = this;
}

template <class _T>
NN_List<_T>::~NN_List() {
	for (NN_List* p_list : _p_list) {
		if (p_list != this) delete p_list;
	}
}

template <class _T>
const _T& NN_List<_T>::operator=(const _T& val) {
	if (_p_list.front() == this) {
		if (_data) {
			*_data = val;
		}
		else {
			_data = std::shared_ptr<_T>(new _T(val));
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::operator=] Can't substitute value."
			);
		}
		else {
			ErrorExcept(
				"[NN_List<_T>::operator=] Can't substitute scalar to list."
			);
		}
	}

	return val;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator=(const NN_List& p) {
	if (&p == this) return *this;

	if (_p_list.front() == this) {
		if (_data) {
			if (p._p_list.size() > 1) {
				ErrorExcept(
					"[NN_List<_T>::operator=] Scalar can't substitute list."
				);
			}
			else if (p._data == NULL) {
				ErrorExcept(
					"[NN_List<_T>::operator=] Can't substitute none list."
				);
			}
			*_data = *p._data;
		}
		else {
			_p_list.clear();

			_data = p._data;
			_p_list.resize(p._p_list.size(), NULL);

			if (p._p_list.front() != &p) {
				int i = 0;
				for (const NN_List& list : p) _p_list[i++] = new NN_List(list);
			}
			else _p_list.front() = this;
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::operator=] Can't operate this function."
			);
		}
		else {
			if (_p_list.size() != p._p_list.size()) {
				ErrorExcept(
					"[NN_List<_T>::operator=] Different size of list."
				);
			}

			for (size_t i = 0; i < _p_list.size(); ++i) *_p_list[i] = *p._p_list[i];
		}
	}

	return *this;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator=(NN_List&& p) {
	if (&p == this) return *this;

	_data = p._data;
	_p_list = p._p_list;

	if (_p_list.front() == &p) _p_list.front() = this;

	p._data = NULL;
	p._p_list.clear();

	return *this;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator=(const std::initializer_list<_T>& list) {
	if (_p_list.front() == this) {
		if (_data) {
			if (list.size() > 1) {
				ErrorExcept(
					"[NN_List<_T>::operator=] Scalar can't substitute by list."
				);
			}
			*_data = *(list.begin());
		}
		else {
			if (list.size() > 1) {
				int i = 0;
				_p_list.resize(list.size(), NULL);

				for (const _T& elem : list) _p_list[i++] = new NN_List(elem);
			}
			else *_data = *(list.begin());
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::operator=] Can't operate this function."
			);
		}
		else {
			if (_p_list.size() != list.size()) {
				ErrorExcept(
					"[NN_List<_T>::operator=] Different size of list."
				);
			}

			int i = 0;
			for (const _T& elem : list) *_p_list[i++] = elem;
		}
	}

	return *this;
}

template <class _T>
NN_List<_T>& NN_List<_T>::operator[](int index) {
	return *_p_list[index];
}

template <class _T>
void NN_List<_T>::clear() {
	_data.reset();

	for (NN_List* p_list : _p_list) {
		if (p_list != this) delete p_list;
	}

	_p_list.clear();
	_p_list.push_back(this);
}

template <class _T>
void NN_List<_T>::append(const _T& val) {
	/*
	_data(x) _p_list.front() != this : list
	_data(x) _p_list.front() == this : none
	_data(o) _p_list.front() != this : invalid
	_data(o) _p_list.front() == this : scalar
	*/

	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_p_list.push_back(new NN_List(val));
			_data = NULL;
		}
		else {
			_data = std::shared_ptr<_T>(new _T(val));
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::append] Can't append this value."
			);
		}
		else {
			_p_list.push_back(new NN_List(val));
		}
	}
}

template <class _T>
void NN_List<_T>::append(_T&& val) {
	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_p_list.push_back(new NN_List(val));
			_data = NULL;
		}
		else {
			_data = std::shared_ptr<_T>(new _T(val));
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::append] Can't append this value."
			);
		}
		else {
			_p_list.push_back(new NN_List(val));
		}
	}
}

template <class _T>
void NN_List<_T>::append(const std::initializer_list<_T>& list) {
	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_p_list.push_back(new NN_List(list));
			_data = NULL;
		}
		else {
			int i = 0;
			_p_list.resize(list.size(), NULL);

			for (const _T& elem : list) _p_list[i++] = new NN_List(elem);
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::append] Can't append this list."
			);
		}
		else {
			_p_list.push_back(new NN_List(list));
		}
	}
}

template <class _T>
void NN_List<_T>::append(const NN_List& list) {
	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_p_list.push_back(new NN_List(list));
			_data = NULL;
		}
		else {
			int i = 0;
			_p_list.resize(list.size(), NULL);

			for (const NN_List& m_list : list) _p_list[i++] = new NN_List(m_list);
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::append] Can't append this list."
			);
		}
		else {
			_p_list.push_back(new NN_List(list));
		}
	}
}

template <class _T>
void NN_List<_T>::extend(const _T& val) {
	/*
	_data(x) _p_list.front() != this : list
	_data(x) _p_list.front() == this : none
	_data(o) _p_list.front() != this : invalid
	_data(o) _p_list.front() == this : scalar
	*/

	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_p_list.push_back(new NN_List(val));
			_data = NULL;
		}
		else {
			_data = std::shared_ptr<_T>(new _T(val));
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::extend] Can't extend this value."
			);
		}
		else {
			_p_list.push_back(new NN_List(val));
		}
	}
}

template <class _T>
void NN_List<_T>::extend(_T&& val) {
	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_p_list.push_back(new NN_List(val));
			_data = NULL;
		}
		else {
			_data = std::shared_ptr<_T>(new _T(val));
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::extend] Can't extend this value."
			);
		}
		else {
			_p_list.push_back(new NN_List(val));
		}
	}
}

template <class _T>
void NN_List<_T>::extend(const std::initializer_list<_T>& list) {
	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_data = NULL;
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::extend] Can't extend this value."
			);
		}
	}

	for (const _T& elem : list) _p_list.push_back(new NN_List(elem));
}

template <class _T>
void NN_List<_T>::extend(const NN_List& list) {
	if (_p_list.front() == this) {
		if (_data) {
			_p_list.front() = new NN_List(_data);
			_data = NULL;
		}
	}
	else {
		if (_data) {
			ErrorExcept(
				"[NN_List<_T>::extend] Can't extend this value."
			);
		}
	}

	for (const NN_List& m_list : list) _p_list.push_back(new NN_List(m_list));
}

template <class _T>
_T& NN_List<_T>::val() {
	return *_data;
}

template <class _T>
size_t NN_List<_T>::size() {
	size_t size = 0;

	if (_p_list.front() == this && _data == NULL) size = 0;
	else size = _p_list.size();

	return size;
}

template <class _T>
void NN_List<_T>::reserve(size_t size) {
	clear();

	_p_list.resize(size, NULL);

	for (NN_List*& p_list : _p_list) p_list = new NN_List;
}

template <class _T>
void NN_List<_T>::resize(size_t size) {
	clear();

	_p_list.resize(size, NULL);

	for (NN_List*& p_list : _p_list) p_list = new NN_List(_T());
}

template <class _T>
size_t NN_List<_T>::size() const {
	size_t size = 0;

	if (_p_list.front() == this && _data == NULL) size = 0;
	else size = _p_list.size();

	return size;
}

template <class _T>
const _T& NN_List<_T>::val() const {
	return *_data;
}

template <class _T>
const NN_List<_T>& NN_List<_T>::operator[](int index) const {
	return *_p_list[index];
}

template <class _T>
typename NN_List<_T>::Iterator NN_List<_T>::begin() {
	return NN_List<_T>::Iterator(_p_list.begin());
}

template <class _T>
typename NN_List<_T>::Iterator NN_List<_T>::end() {
	return NN_List<_T>::Iterator(_p_list.end());
}

template <class _T>
typename NN_List<_T>::ConstIterator NN_List<_T>::begin() const {
	return NN_List<_T>::ConstIterator(_p_list.cbegin());
}

template <class _T>
typename NN_List<_T>::ConstIterator NN_List<_T>::end() const {
	return NN_List<_T>::ConstIterator(_p_list.cend());
}

#ifdef PUT_LIST
template <class _T>
std::ostream& NN_List<_T>::put_list(std::ostream& os) const {
	os << '[';

	for (const NN_List* list : _p_list) {
		if (list->_p_list.front() == list) os << list->val();
		else list->put_list(os);

		os << ", ";
	}

	os << ']';

	return os;
}

template <class _T>
std::ostream& operator<<(std::ostream& os, const NN_List<_T>& list) {
	return list.put_list(os) << std::endl;
}
#endif
