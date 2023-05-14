#pragma once
#include <vector>
#include <iostream>
//#include <opencv2/opencv.hpp>
#include "CudaCheck.h"
#include "ptrManager.h"


#define STR_MAX			1024

#define SMALL_XY				16
#define SMALL_Z					4
#define BLOCK_SIZE				32
#define SQR_BLOCK_SIZE			BLOCK_SIZE * BLOCK_SIZE
#define CONST_ELEM_SIZE			(65536 / sizeof(uint))

#define EPSILON					1e-8

#define STREAMS					32

#define FIX_MODE

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

typedef std::vector<int> nn_shape;

const char* dimension_to_str(const nn_shape& shape);
dim3 get_grid_size(const dim3 block, unsigned int x = 1, unsigned int y = 1, unsigned int z = 1);

class NN_Shared_Ptr {
protected:
	ptrRef* id;
	static ptrManager linker;

public:
	NN_Shared_Ptr();
};

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
protected:
	static List<_T>* create_head();
	static void clear_head(List<_T>** head);
	static void insert_link(List<_T>* current, List<_T>* prev_node);
	static void clear_link(List<_T>* current);

public:
	List* _prev;
	List* _next;
	List* _head;

	_T _val;

	bool _is_scalar;
	size_t _size;

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
	size_t size();

#ifdef PUT_LIST
	void put(std::ostream& os) const;
#endif
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
void List<_T>::clear_head(List<_T>** head) {
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
	_is_scalar(false),
	_size(0)
{
	id = NULL;
}

template <class _T>
List<_T>::List(const _T& val) :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_is_scalar(true),
	_size(1)
{
	this->_val = val;

	id = NULL;
}

template <class _T>
List<_T>::List(const std::initializer_list<_T>& list) :
	_prev(NULL),
	_next(NULL),
	_is_scalar(false),
	_size(0)
{
	_head = create_head();

	for (const _T& val : list) {
		List<_T>* current = new List<_T>(val);
		List<_T>::insert_link(current, _head->_prev);
		++_size;
	}

	id = linker.create();
}

template <class _T>
List<_T>::List(const std::initializer_list<List>& list) :
	_prev(NULL),
	_next(NULL),
	_is_scalar(false),
	_size(0)
{
	_head = create_head();

	for (const List<_T>& p_list : list) {
		List<_T>* current = new List<_T>(p_list);
		List<_T>::insert_link(current, _head->_prev);
		++_size;
	}

	id = linker.create();
}

template <class _T>
List<_T>::List(const List& p) :
	_prev(NULL),
	_next(NULL),
	_val(p._val),
	_head(p._head),
	_is_scalar(p._is_scalar),
	_size(p._size)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <class _T>
List<_T>::List(List&& p) :
	_prev(NULL),
	_next(NULL),
	_val(p._val),
	_head(p._head),
	_is_scalar(p._is_scalar),
	_size(p._size)
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
			clear_head(&_head);
			linker.erase(&id);
		}
	}

	_val = _T();
	_is_scalar = false;
	_size = 0;
	_head = NULL;
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	clear_head(&_head);

	_val = p._val;
	_head = p._head;
	_is_scalar = p._is_scalar;
	_size = p._size;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator=(List&& p) {
	clear_head(&_head);

	_val = p._val;
	_head = p._head;
	_is_scalar = p._is_scalar;
	_size = p._size;

	id = p.id;

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
	if (_head == NULL) {
		if (!_is_scalar) {
			_val = val;
			_is_scalar = true;
		}
		else {
			_head = create_head();
			insert_link(new List<_T>(_val), _head->_prev);

			_val = _T();
			_is_scalar = false;

			insert_link(new List<_T>(val), _head->_prev);

			id = linker.create();
		}
	}
	else insert_link(new List<_T>(val), _head->_prev);
	++_size;
}

template <class _T>
size_t List<_T>::size() {
	return _size;
}

#ifdef PUT_LIST
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
#endif
