#pragma once
#include <vector>
#include <iostream>
#include "CudaCheck.h"
#include "ObjectID.h"


typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

#define STR_MAX			1024

#define BLOCK_SIZE				32
#define SQR_BLOCK_SIZE			BLOCK_SIZE * BLOCK_SIZE
#define CONST_ELEM_SIZE			16384
#define CONST_MEM_SIZE			65536

#define EPSILON					1e-8

#define FIX_MODE				true


dim3 get_grid_size(const dim3 block, uint x = 1, uint y = 1, uint z = 1);

class NN_Shared_Ptr {
protected:
	Object_ID* id;
	static Object_Linker linker;

public:
	NN_Shared_Ptr();
};

struct NN_Tensor4D {
	float* data;

	int n;
	int c;
	int h;
	int w;
};

const size_t get_elem_size(const NN_Tensor4D& tensor);

/**********************************************/
/*                                            */
/*                     List                   */
/*                                            */
/**********************************************/

template <class _T>
class List : public NN_Shared_Ptr {
protected:
	static List<_T>* create_head();
	static void clear(List<_T>** head);
	static void insert_link(List<_T>* current, List<_T>* prev_node);
	static void clear_link(List<_T>* current);

public:
	List* _prev;
	List* _next;
	List* _head;

	_T _val;

	bool _is_scalar;

	class Iterator {
	public:
		bool scalar_switch;
		const List<_T>* _this;
		List<_T>* _current;

		Iterator(const List<_T>* p_this, List<_T>* current);
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

	List<_T>& operator[](const int index);

	const Iterator begin() const;
	const Iterator end() const;

	_T& get();
	void push_back(const _T& val);
	//void put(std::ostream& os) const;
};

template <class _T>
List<_T>::Iterator::Iterator(const List<_T>* p_this, List<_T>* current) :
	_this(p_this),
	_current(current)
{
	scalar_switch = _current == NULL ? true : false;
}

template <class _T>
List<_T>::Iterator::Iterator(typename const List<_T>::Iterator& p) :
	_this(p._this),
	_current(p._current),
	scalar_switch(p.scalar_switch)
{
}

template <class _T>
void List<_T>::Iterator::operator++() {
	if (_current == NULL) scalar_switch = false;
	else _current = _current->_next;
}

template <class _T>
bool List<_T>::Iterator::operator!=(typename const List<_T>::Iterator& p) const {
	if (_current == NULL) return scalar_switch;
	
	return _current != p._current;
}

template <class _T>
const List<_T>& List<_T>::Iterator::operator*() const {
	if (_current == NULL) return *_this;

	return *_current;
}

template <class _T>
List<_T>* List<_T>::create_head() {
	List<_T>* nodes = new List<_T>();

	nodes->_prev = nodes;
	nodes->_next = nodes;

	return nodes;
}

template <class _T>
void List<_T>::clear(List<_T>** head) {
	if (*head == NULL) return;

	List<_T>* current = (*head)->_next;
	List<_T>* tmp = NULL;

	while (current != *head) {
		tmp = current->_next;
		delete current;
		current = tmp;
	}
	
	delete *head;
	*head = NULL;
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
	_is_scalar(false)
{
	id = NULL;
}

template <class _T>
List<_T>::List(const _T& val) :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_is_scalar(true)
{
	this->_val = val;

	id = NULL;
}

template <class _T>
List<_T>::List(const std::initializer_list<_T>& list) :
	_prev(NULL),
	_next(NULL),
	_is_scalar(false)
{
	_head = create_head();

	for (const _T& val : list) {
		List<_T>* current = new List<_T>(val);
		List<_T>::insert_link(current, _head->_prev);
	}

	id = linker.Create();
}

template <class _T>
List<_T>::List(const std::initializer_list<List>& list) :
	_prev(NULL),
	_next(NULL),
	_is_scalar(false)
{
	_head = create_head();

	for (const List<_T>& p_list : list) {
		List<_T>* current = new List<_T>(p_list);
		List<_T>::insert_link(current, _head->_prev);
	}

	id = linker.Create();
}

template <class _T>
List<_T>::List(const List& p) :
	_prev(NULL),
	_next(NULL)
{
	_val = p._val;
	_head = p._head;

	id = p.id;

	if (id) ++id->ref_cnt;
}

template <class _T>
List<_T>::List(List&& p) :
	_prev(NULL),
	_next(NULL) 
{
	_val = p._val;
	_head = p._head;

	id = linker.Create();

	p._head = NULL;
	p.id = NULL;
}

template <class _T>
List<_T>::~List() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			clear(&_head);
			linker.Erase(id);
		}
	}

	_head = NULL;
	id = NULL;
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	clear(&_head);

	_val = p._val;
	_head = p._head;
	
	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator=(List&& p) {
	clear(&_head);

	_val = p._val;
	_head = p._head;

	p._head = NULL;
	p.id = NULL;

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator[](const int index) {
	List<_T>* p_element = NULL;

	if (_head == NULL) {
		if (index != 0) {
			ErrorExcept(
				"[List::operator[]] this param is scalar"
			);
		}
		p_element = this;
	}
	else if (index < 0) {
		ErrorExcept(
			"[List::operator[]] invalid index (%d).",
			index
		);
	}
	else {
		List<_T>* current = _head->_next;
		for (int i = 0; i < index; ++i) {
			current = current->_next;

			if (current == _head) {
				ErrorExcept(
					"[List::operator[]] this list size are %d. but index is %d.",
					i + 1, index
				);
			}
		}

		p_element = current;
	}

	return *p_element;
}

template <class _T>
typename const List<_T>::Iterator List<_T>::begin() const {
	if (_head == NULL) return List<_T>::Iterator(this, NULL);

	return List<_T>::Iterator(this, _head->_next);
}

template <class _T>
typename const List<_T>::Iterator List<_T>::end() const {
	return List<_T>::Iterator(this, _head);
}

template <class _T>
_T& List<_T>::get() {
	if (_head != NULL) {
		ErrorExcept("[List::get()] this element is not scalar.");
	}

	return _val;
}

template <class _T>
void List<_T>::push_back(const _T& val) {
	List<_T>* current = NULL;

	if (_head == NULL) {
		_head = create_head();

		if (_is_scalar) {
			current = new List<_T>(this->_val);
			insert_link(current, _head->_prev);
			
			_val = _T();
			_is_scalar = false;
		}
	}

	current = new List<_T>(val);
	insert_link(current, _head->_prev);
}

/*
template <class _T>
void List<_T>::put(std::ostream& os) const {
	if (_head == NULL) {
		os << _val << ", ";
	}
	else {
		os << '[';

		List<_T>* current = _head->_next;

		while (current != _head) {
			current->put(os);
			current = current->_next;
		}

		os << "], ";
	}
}

template <class _T>
std::ostream& operator<<(std::ostream& os, const List<_T>& list) {
	list.put(os);
	os << std::endl;

	return os;
}
*/