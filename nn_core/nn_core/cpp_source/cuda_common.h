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
class List : public NN_Shared_Ptr {
	static uint _count_create;
protected:
	static void init_head(List<_T>* head);
	static void clear_head(List<_T>* head);
	static void insert_link(List<_T>* current, List<_T>* prev_node);
	static void clear_link(List<_T>* current);

public:
	List* _prev;
	List* _next;
	List* _head;

	_T* _val;

	uint _len;

	class Iterator {
	public:
		const List<_T>* _current;

		Iterator(const List<_T>* current);
		Iterator(typename const Iterator& p);

		void operator++();
		bool operator!=(typename const Iterator& p) const;
		const List<_T>& operator*() const;
	};

	List();
	List(const _T& val);
	List(const std::initializer_list<_T>& list);
	List(const std::initializer_list<List>& list);
	List(const List& p);
	List(List&& p);
	~List();

	List& operator=(const List& p);
	List& operator=(List&& p);

	const Iterator begin() const;
	const Iterator end() const;

	_T& get();
	void push_back(const _T& val);
	uint size();
};

template <class _T>
uint List<_T>::_count_create = 0;

template <class _T>
List<_T>::Iterator::Iterator(const List<_T>* _this) :
	_current(_this->_head ? _this->_head : _this)
{
}

template <class _T>
List<_T>::Iterator::Iterator(typename const List<_T>::Iterator& p) :
	_current(p._current)
{
}

template <class _T>
void List<_T>::Iterator::operator++() {
	_current = _current->_next;
}

template <class _T>
bool List<_T>::Iterator::operator!=(typename const List<_T>::Iterator& p) const {
	if (_current) return _current != p._current;
	else return false;
}

template <class _T>
const List<_T>& List<_T>::Iterator::operator*() const {

	return *_current;
}

template <class _T>
void List<_T>::init_head(List<_T>* head) {
	head->_prev = head;
	head->_next = head;
	head->_head = head;
}

template <class _T>
void List<_T>::clear_head(List<_T>* head) {
	if (head == NULL) return;

	List<_T>* current = head->_next;
	List<_T>* tmp = NULL;

	while (current != head) {
		tmp = current->_next;
		delete current->_val;
		delete current;
		current = tmp;
	}
}

template <class _T>
void List<_T>::insert_link(List<_T>* current, List<_T>* prev_node) {
	List<_T>* before = prev_node;
	List<_T>* after = prev_node->_next;

	before->_next = current;
	after->_prev = current;

	current->_next = after;
	current->_prev = before;
}

template <class _T>
void List<_T>::clear_link(List<_T>* current) {
	List<_T>* before = current->_prev;
	List<_T>* after = current->_next;

	before->_next = after;
	after->_prev = before;

	current->_next = NULL;
	current->_prev = NULL;
}


template <class _T>
List<_T>::List() :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_val(NULL),
	_len(0)
{
	++_count_create;

	if (_count_create == 1) {
		init_head(this);
		id = linker.create();
	}
	else id = NULL;

	--_count_create;
}

template <class _T>
List<_T>::List(const _T& val) :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_val(NULL),
	_len(0)
{
	++_count_create;

	_val = new _T(val);

	if (_count_create == 1) {
		init_head(this);
		id = linker.create();
	}
	else id = NULL;

	--_count_create;
}

template <class _T>
List<_T>::List(const std::initializer_list<_T>& list) :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_val(NULL),
	_len(0)
{
	++_count_create;

	init_head(this);

	for (const _T& val : list) {
		List<_T>* current = new List<_T>(val);

		List<_T>::insert_link(current, _head->_prev);
		++_len;
	}

	id = linker.create();

	--_count_create;
}

template <class _T>
List<_T>::List(const std::initializer_list<List>& list) :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_val(NULL),
	_len(0)
{
	++_count_create;

	init_head(this);

	for (const List<_T>& p_list : list) {
		List<_T>* current = new List<_T>(p_list);
		List<_T>::insert_link(current, _head->_prev);
		++_len;
	}

	id = linker.create();

	--_count_create;
}

template <class _T>
List<_T>::List(const List& p) :
	_prev(p._prev),
	_next(p._next),
	_val(p._val),
	_head(p._head),
	_len(p._len)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <class _T>
List<_T>::List(List&& p) :
	_prev(p._prev),
	_next(p._next),
	_val(p._val),
	_head(p._head),
	_len(p._len)
{
	id = p.id;

	p._head = NULL;
	p.id = NULL;
}

template <class _T>
List<_T>::~List() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			clear_head(_head);
			//delete _head->_val;
			//delete _head;
			
			linker.erase(id);
		}
	}
	_head = NULL;
	_val = NULL;
	_len = 0;
	id = NULL;
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	clear_head(_head);

	_val = p._val;
	_head = p._head;
	_prev = NULL;
	_next = NULL;
	_len = p._len;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator=(List&& p) {
	clear_head(_head);

	_val = p._val;
	_head = p._head;
	_prev = NULL;
	_next = NULL;
	_len = 0;

	id = p.id;

	p._head = NULL;
	p.id = NULL;

	return *this;
}

template <class _T>
typename const List<_T>::Iterator List<_T>::begin() const {
	return List<_T>::Iterator(_head);
}

template <class _T>
typename const List<_T>::Iterator List<_T>::end() const {
	return List<_T>::Iterator(_next);
}

template <class _T>
_T& List<_T>::get() {
	if (_head != NULL) {
		ErrorExcept("[List::get()] this element is not scalar.");
	}

	return *_val;
}

template <class _T>
void List<_T>::push_back(const _T& val) {
	++_count_create;

	List<_T>* node = new List<_T>(val);

	insert_link(node, _head->_prev);

	--_count_create;
}

template <class _T>
uint List<_T>::size() {
	return _len;
}