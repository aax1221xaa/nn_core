#pragma once
#include <vector>
#include <iostream>
#include "CudaCheck.h"


#define STR_MAX					1024

#define BLOCK_4					4
#define BLOCK_8					8
#define BLOCK_16				16
#define BLOCK_32				32
#define BLOCK_1024				1024
#define CONST_ELEM_SIZE			(65536 / sizeof(uint))

#define EPSILON					1e-8

#define STREAMS					32

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

dim3 get_grid_size(const dim3 block, unsigned int x = 1, unsigned int y = 1, unsigned int z = 1);


/*******************************************

					 Pad

*******************************************/

enum class Pad { VALID, SAME };

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
protected:
	struct Container {
		uint n_ref;
		_T* data;
	};

	static void insert_node(List* prev, List* node);
	static void erase_node(List** node);
	static void destroy_node(List* head);
	static void put_list(std::ostream& os, List* list);

public:
	Container* _ptr;

	List* _prev;
	List* _next;
	List* _head;

	class Iterator {
	public:
		List* _node;

		Iterator(List* node);
		Iterator(typename const Iterator& iter);

		void operator++();
		bool operator!=(typename const Iterator& p) const;
		List& operator*() const;
	};

	List();
	List(const _T& val);
	List(const std::initializer_list<_T>& list);
	List(const std::initializer_list<List>& list);
	List(const List& p);
	~List();

	List& operator=(const List& p);
	List& operator[](uint index);

	void clear();
	void push_back(const _T& val);
	void push_back(const List<_T>& list);
	_T& get_val() const;
	void put(std::ostream& os);

	typename Iterator begin() const;
	typename Iterator end() const;
};

std::ostream& operator<<(std::ostream& os, List<int>& list);

template <class _T>
void List<_T>::insert_node(List<_T>* prev, List<_T>* node) {
	List<_T>* before = prev;
	List<_T>* after = prev->_next;

	before->_next = node;
	after->_prev = node;

	node->_prev = before;
	node->_next = after;
}

template <class _T>
void List<_T>::erase_node(List<_T>** node) {
	List<_T>* before = (*node)->_prev;
	List<_T>* after = (*node)->_next;

	before->_next = after;
	after->_prev = before;

	delete *node;
	*node = NULL;
}

template <class _T>
void List<_T>::destroy_node(List<_T>* head) {
	List<_T>* node = head->_next;
	List<_T>* tmp = NULL;

	while (node != head) {
		tmp = node->_next;
		delete node;
		node = tmp;
	}
}

template <class _T>
void List<_T>::put_list(std::ostream& os, List<_T>* list) {
	if (list->_ptr) os << list->get_val() << ", ";
	else if (list->_head != list) {
		os << '[';
		
		List<_T>* tmp = list->_head->_next;
		while (tmp != list->_head) {
			put_list(os, tmp);
			tmp = tmp->_next;
		}

		os << ']';
	}
}

template <class _T>
List<_T>::Iterator::Iterator(List<_T>* node) :
	_node(node)
{
}

template <class _T>
List<_T>::Iterator::Iterator(typename const List<_T>::Iterator& iter) :
	_node(iter._node)
{
}

template <class _T>
void List<_T>::Iterator::operator++() {
	if (_node == _node->_next) _node = NULL;
	else _node = _node->_next;
}

template <class _T>
bool List<_T>::Iterator::operator!=(typename const List<_T>::Iterator& p) const {
	if (_node) {
		if (_node == _node->_next) return true;
		else return _node != p._node;
	}
	else return false;
}

template <class _T>
List<_T>& List<_T>::Iterator::operator*() const {
	return *_node;
}

template <class _T>
List<_T>::List() :
	_prev(this),
	_next(this),
	_head(this),
	_ptr(NULL)
{
}

template <class _T>
List<_T>::List(const _T& val) :
	_prev(this),
	_next(this),
	_head(this)
{
	_ptr = new Container;
	_ptr->data = new _T(val);
	_ptr->n_ref = 1;
}

template <class _T>
List<_T>::List(const std::initializer_list<_T>& list) :
	_prev(this),
	_next(this),
	_ptr(NULL)
{
	_head = new List<_T>;

	for (const _T& val : list) 
		insert_node(_head->_prev, new List<_T>(val));
}

template <class _T>
List<_T>::List(const std::initializer_list<List<_T>>& list) :
	_prev(this),
	_next(this),
	_ptr(NULL)
{
	_head = new List<_T>;

	for (const List<_T>& m_list : list)
		insert_node(_head->_prev, new List<_T>(m_list));
}

template <class _T>
List<_T>::List(const List<_T>& p) :
	_prev(this),
	_next(this),
	_head(this),
	_ptr(NULL)
{
	if (p._ptr) {
		_ptr = p._ptr;
		++_ptr->n_ref;
	}
	else if (p._head != &p) {
		_head = new List<_T>(*p._head);
	}
	else {
		List<_T>* node = p._next;

		while (node != &p) {
			insert_node(_prev, new List<_T>(*node));
			node = node->_next;
		}
	}
}

template <class _T>
List<_T>::~List() {
	clear();
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	clear();

	if (p._ptr) {
		_ptr = p._ptr;
		++_ptr->n_ref;
	}
	else if (p._head != &p) {
		_head = new List<_T>(*p._head);
	}

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator[](uint index) {
	List<_T>* tmp = _head != this ? _head->_next : this;
		
	for (uint i = 0; i < index; ++i) tmp = tmp->_next;

	return *tmp;
}

template <class _T>
void List<_T>::clear() {
	if (_ptr) {
		if (_ptr->n_ref > 1) --_ptr->n_ref;
		else {
			delete _ptr->data;
			delete _ptr;
		}
	}
	else if (_head != this) {
		delete _head;
	}
	else if (_next != this) {
		destroy_node(this);
	}

	_prev = this;
	_next = this;
	_head = this;
	_ptr = NULL;
}

template <class _T>
void List<_T>::push_back(const _T& val) {
	if (_head == this) _head = new List<_T>;

	insert_node(_head->_prev, new List<_T>(val));
}

template <class _T>
void List<_T>::push_back(const List<_T>& list) {
	if (_head == this) _head = new List<_T>;

	insert_node(_head->_prev, new List<_T>(list));
}

template <class _T>
_T& List<_T>::get_val() const {
	if (!_ptr) {
		ErrorExcept("[List<_T>::get_val] Can't get value.");
	}

	return *(_ptr->data);
}

template <class _T>
void List<_T>::put(std::ostream& os) {
	put_list(os, this);
	os << std::endl;
}

template <class _T>
typename List<_T>::Iterator List<_T>::begin() const {
	return List<_T>::Iterator(_head->_next);
}

template <class _T>
typename List<_T>::Iterator List<_T>::end() const {
	return List<_T>::Iterator(_head);
}


/**********************************************/
/*                                            */
/*                   NN_Shape                 */
/*                                            */
/**********************************************/

class NN_Shape {
private:
	class _Container {
	public:
		int* _dims;
		int _n_ref;

		_Container() : _dims(NULL), _n_ref(0) {}
	};

	_Container* _data;
	int _len;

public:
	class Iterator {
	public:
		int* _p_dims;
		int _index;

		Iterator(int* p_dims, int index);
		Iterator(const typename Iterator& p);

		void operator++();
		bool operator!=(const typename Iterator& p) const;
		int& operator*() const;
	};

	NN_Shape();
	NN_Shape(int len);
	NN_Shape(const std::initializer_list<int>& dims);
	NN_Shape(const NN_Shape& p);
	NN_Shape(NN_Shape&& p);
	~NN_Shape();

	NN_Shape& operator=(const NN_Shape& p);
	NN_Shape& operator=(NN_Shape&& p);

	int& operator[](const int& index) const;

	void clear();
	void set(const std::initializer_list<int>& dims);
	void resize(int len);
	void push_back(const int dim);
	void push_front(const int dim);
	const int& get_size() const;
	int* get_dims() const;
	void copy_to(NN_Shape& shape) const;

	typename Iterator begin() const;
	typename Iterator end() const;
};

const char* put_shape(const NN_Shape& tensor);