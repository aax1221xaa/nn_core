#pragma once
#include <vector>
#include <iostream>
//#include <opencv2/opencv.hpp>
#include "CudaCheck.h"
#include "ptrManager.h"


#define STR_MAX					1024

#define BLOCK_4					4
#define BLOCK_8					8
#define BLOCK_16				16
#define BLOCK_32				32
#define BLOCK_1024				1024
#define CONST_ELEM_SIZE			(65536 / sizeof(uint))

#define EPSILON					1e-8

#define STREAMS					32

#define DIM_SIZE				4

#define FIX_MODE

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

dim3 get_grid_size(const dim3 block, unsigned int x = 1, unsigned int y = 1, unsigned int z = 1);

class NN_Shared_Ptr {
protected:
	ptrRef* id;
	static ptrManager linker;

public:
	NN_Shared_Ptr();
};

/*******************************************

					 Pad

*******************************************/

enum class Pad { VALID, SAME };

/**********************************************/
/*                                            */
/*                   nn_shape                 */
/*                                            */
/**********************************************/

typedef std::vector<int> nn_shape;

const char* put_shape(const nn_shape& tensor);

/**********************************************/
/*                                            */
/*                   nn_type                  */
/*                                            */
/**********************************************/

typedef float nn_type;

/**********************************************/
/*                                            */
/*                     List                   */
/*                                            */
/**********************************************/

template <class _T>
class List {
public:
	std::vector<List> _list;

	_T _val;

	List();
	List(const _T& val);
	List(const std::initializer_list<_T>& list);
	List(const std::initializer_list<List>& list);
	List(const List& p);

	List& operator=(const List& p);
	List& operator[](int index);

	void push_back(const _T& val);

	class Iterator {
	protected:
		bool _switch;
		List& _this;

	public:
		typename std::vector<List>::iterator _iter;

		Iterator(List* const curr_this, typename std::vector<List>::iterator current_iter, bool is_scalar);

		void operator++();
		bool operator!=(typename const Iterator& p);
		List<_T>& operator*();
	};

	typename Iterator begin();
	typename Iterator end();
};

template <class _T>
List<_T>::List() :
	_val(_T())
{
}

template <class _T>
List<_T>::List(const _T& val) :
	_val(val)
{
}

template <class _T>
List<_T>::List(const std::initializer_list<_T>& list) :
	_val(_T())
{
	for (const _T& val : list) _list.push_back(val);
}

template <class _T>
List<_T>::List(const std::initializer_list<List>& list) :
	_val(_T())
{
	for (const List<_T>& m_list : list) {
		if (m_list._list.size() > 0) _list.push_back(m_list);
		else _list.push_back(m_list._val);
	}
}

template <class _T>
List<_T>::List(const List<_T>& p) :
	_list(p._list),
	_val(p._val)
{
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	_list = p._list;
	_val = p._val;

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator[](int index) {
	index = abs(index);

	if (_list.size() > 0) {
		if (index >= (int)_list.size()) {
			ErrorExcept(
				"[List<_T>::operator[]]: Index was overflowed."
			);
		}

		return _list[index];
	}
	else {
		if (index > 0) {
			ErrorExcept(
				"[List<_T>::operator[]]: Index was overflowed."
			);
		}

		return *this;
	}
}

template <class _T>
void List<_T>::push_back(const _T& val) {
	_list.push_back(val);
}

template <class _T>
List<_T>::Iterator::Iterator(List* const curr_this, typename std::vector<List<_T>>::iterator current_iter, bool is_scalar) :
	_this(*curr_this),
	_iter(current_iter),
	_switch(is_scalar)
{
}

template <class _T>
void List<_T>::Iterator::operator++() {
	if (_switch) _switch = false;
	else ++_iter;
}

template <class _T>
bool List<_T>::Iterator::operator!=(typename const Iterator& p) {
	if (_switch) return true;
	else return _iter != p._iter;
}

template <class _T>
List<_T>& List<_T>::Iterator::operator*() {
	if (_switch) return _this;
	else return *_iter;
}

template <class _T>
typename List<_T>::Iterator List<_T>::begin() {
	if (_list.size() > 0) return Iterator(this, _list.begin(), false);
	else return Iterator(this, _list.begin(), true);
}

template <class _T>
typename List<_T>::Iterator List<_T>::end() {
	if (_list.size() > 0) return Iterator(this, _list.end(), false);
	else return Iterator(this, _list.end(), true);
}
