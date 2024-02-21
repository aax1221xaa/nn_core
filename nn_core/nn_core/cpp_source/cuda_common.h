#pragma once
#include <vector>
#include <iostream>
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

//dim3 get_grid_size(const dim3 block, unsigned int x = 1, unsigned int y = 1, unsigned int z = 1);

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
/*                   nn_type                  */
/*                                            */
/**********************************************/

typedef float nn_type;

/**********************************************/
/*                                            */
/*                    Vector                  */
/*                                            */
/**********************************************/

#define ADDR_ALLOC				10

template <class _T>
class Vector : public NN_Shared_Ptr {
public:
	class Node {
	public:
		Node* _prev;
		Node* _next;
		
		_T _val;

		Node() : _prev(NULL), _next(NULL) {}
		Node(const _T& val) : _prev(NULL), _next(NULL), _val(val) {}
	};

	class Iterator {
	public:
		Node* _node;

		Iterator(Node* node);
		Iterator(const Iterator& p);

		void operator++();
		bool operator!=(typename const Iterator& p) const;
		_T& operator*() const;
	};

	static int _count_create;

	Node* _head;
	Node** _ptr_list;

	size_t _ptr_len;
	size_t _n_used;

	static typename Node* create_head();
	static void insert_node(Node* prev, Node* node);
	static void erase_node(Node** node);
	static void destroy_node(Node** head);

	static typename Node** create_ptr_list(size_t len);
	static void resize_ptr_list(Node**& ptr_list, size_t prev_len, size_t current_len);
	
	void clear();

	Vector();
	Vector(const _T& val);
	Vector(const std::initializer_list<_T>& list);
	Vector(const Vector& p);
	Vector(Vector&& p);
	~Vector();

	Vector& operator=(const Vector& p);
	Vector& operator=(Vector&& p);
	_T& operator[](int index) const;

	void push_back(const _T& val);
	const _T pop_front();
	const _T pop_back();
	size_t size() const;

	void resize(size_t size);

	typename Iterator begin() const;
	typename Iterator end() const;
};

template <class _T>
int Vector<_T>::_count_create = 0;

template <class _T>
Vector<_T>::Iterator::Iterator(Node* node) :
	_node(node)
{
}

template <class _T>
Vector<_T>::Iterator::Iterator(const Iterator& p) :
	_node(p._node)
{
}

template <class _T>
void Vector<_T>::Iterator::operator++() {
	_node = _node->_next;
}

template <class _T>
bool Vector<_T>::Iterator::operator!=(typename const Iterator& p) const {
	return _node != p._node;
}

template <class _T>
_T& Vector<_T>::Iterator::operator*() const {
	return _node->_val;
}

template <class _T>
typename Vector<_T>::Node* Vector<_T>::create_head() {
	Node* head = new Node;

	head->_prev = head;
	head->_next = head;

	return head;
}

template <class _T>
void Vector<_T>::insert_node(Node* prev, Node* node) {
	if (prev == NULL || node == NULL) return;

	Node* before = prev;
	Node* after = prev->_next;

	node->_prev = before;
	node->_next = after;
	before->_next = node;
	after->_prev = node;
}

template <class _T>
void Vector<_T>::erase_node(Node** node) {
	if (node == NULL || *node == NULL) return;

	Node* before = (*node)->_prev;
	Node* after = (*node)->_next;

	before->_next = after;
	after->_prev = before;

	delete *node;

	*node = NULL;
}

template <class _T>
void Vector<_T>::destroy_node(Node** head) {
	if (head == NULL || *head == NULL) return;

	Node* node = (*head)->_next;
	Node* tmp = NULL;

	while (node != *head) {
		tmp = node->_next;
		delete node;
		node = tmp;
	}

	delete *head;
	*head = NULL;
}

template <class _T>
typename Vector<_T>::Node** Vector<_T>::create_ptr_list(size_t len) {
	Node** tmp = new Node*[len];

	memset(tmp, 0, sizeof(Node**) * len);

	return tmp;
}

template <class _T>
void Vector<_T>::resize_ptr_list(Node**& ptr_list, size_t prev_len, size_t current_len) {
	if (prev_len == current_len) return;

	Node** tmp = new Node*[current_len];

	memset(tmp, 0, sizeof(Node**) * current_len);
	
	for (size_t i = 0; i < prev_len; ++i) tmp[i] = ptr_list[i];

	delete[] ptr_list;

	ptr_list = tmp;
}

template <class _T>
void Vector<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			destroy_node(&_head);
			delete[] _ptr_list;

			linker.erase(id);
		}
	}

	_ptr_len = 0;
	_ptr_list = NULL;
	_n_used = 0;
	_head = NULL;

	id = NULL;
}

template <class _T>
Vector<_T>::Vector() :
	_head(NULL),
	_ptr_list(NULL),
	_ptr_len(0),
	_n_used(0)
{	
	++_count_create;

	if (_count_create == 1) {
		_head = create_head();
		id = linker.create();
	}

	--_count_create;
}

template <class _T>
Vector<_T>::Vector(const _T& val) :
	_ptr_list(NULL),
	_ptr_len(0),
	_n_used(0)
{
	++_count_create;
	
	Node* node = new Node(val);

	if (_count_create == 1) {
		_n_used = 1;
		_ptr_len = ADDR_ALLOC;

		_head = create_head();
		insert_node(_head->_prev, node);

		_ptr_list = create_ptr_list(_ptr_len);
		_ptr_list[0] = node;

		id = linker.create();
	}
	else _head = node;

	--_count_create;
}

template <class _T>
Vector<_T>::Vector(const std::initializer_list<_T>& list) {
	++_count_create;

	_ptr_len = ADDR_ALLOC + (list.size() / ADDR_ALLOC) * ADDR_ALLOC;
	_ptr_list = create_ptr_list(_ptr_len);
	_n_used = 0;

	_head = create_head();

	for (const _T& val : list) {
		Node* node = new Node(val);
		insert_node(_head->_prev, node);

		_ptr_list[_n_used++] = node;
	}

	id = linker.create();

	--_count_create;
}

template <class _T>
Vector<_T>::Vector(const Vector<_T>& p) :
	_ptr_len(p._ptr_len),
	_ptr_list(p._ptr_list),
	_n_used(p._n_used),
	_head(p._head)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <class _T>
Vector<_T>::Vector(Vector<_T>&& p) :
	_ptr_len(p._ptr_len),
	_ptr_list(p._ptr_list),
	_n_used(p._n_used),
	_head(p._head)
{
	id = p.id;

	p._ptr_list = NULL;
	p._head = NULL;
	p.id = NULL;
}

template <class _T>
Vector<_T>::~Vector() {
	clear();
}

template <class _T>
Vector<_T>& Vector<_T>::operator=(const Vector<_T>& p) {
	if (this == &p) return *this;

	clear();

	_ptr_len = p._ptr_len;
	_ptr_list = p._ptr_list;
	_n_used = p._n_used;
	_head = p._head;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
Vector<_T>& Vector<_T>::operator=(Vector<_T>&& p) {
	clear();

	_ptr_len = p._ptr_len;
	_ptr_list = p._ptr_list;
	_n_used = p._n_used;
	_head = p._head;

	id = p.id;

	p._ptr_list = NULL;
	p._head = NULL;
	p.id = NULL;

	return *this;
}

template <class _T>
_T& Vector<_T>::operator[](int index) const {
	int m_index = index < 0 ? (int)_n_used + index : index;

	if (!_ptr_list || m_index < 0 || m_index >= _n_used) {
		ErrorExcept(
			"[Vector<_T>::operator()] can't pick %ds element.",
			index
		);
	}

	return _ptr_list[m_index]->_val;
}

template <class _T>
void Vector<_T>::push_back(const _T& val) {
	++_count_create;

	if (id->ref_cnt > 1) {
		Node** prev_ptr_list = _ptr_list;
		size_t prev_ptr_len = _ptr_len;
		size_t prev_n_used = _n_used;

		clear();
		_head = create_head();

		_ptr_len = prev_ptr_len;
		_n_used = prev_n_used;

		if (_ptr_len < _n_used + 1) _ptr_len += ADDR_ALLOC;

		_ptr_list = create_ptr_list(_ptr_len);

		for (size_t i = 0; i < _n_used; ++i) {
			Node* node = new Node(prev_ptr_list[i]->_val);
			insert_node(_head->_prev, node);
			_ptr_list[i] = node;
		}

		id = linker.create();
	}
	else {
		if (_ptr_len < _n_used + 1) {
			_ptr_len += ADDR_ALLOC;

			Node** tmp = create_ptr_list(_ptr_len);

			for (size_t i = 0; i < _n_used; ++i) {
				tmp[i] = _ptr_list[i];
			}

			delete[] _ptr_list;
			_ptr_list = tmp;
		}
	}

	Node* node = new Node(val);

	insert_node(_head->_prev, node);
	_ptr_list[_n_used++] = node;

	--_count_create;
}

template <class _T>
const _T Vector<_T>::pop_front() {
	if (!_head || _head->_next == _head) return _T();
	
	const _T val = _head->_next->_val;
	Node* node = _head->_next;

	erase_node(&node);

	for (size_t i = 1; i < _n_used; ++i) {
		_ptr_list[i - 1] = _ptr_list[i];
	}

	_ptr_list[_n_used] = NULL;
	--_n_used;

	return val;
}

template <class _T>
const _T Vector<_T>::pop_back() {
	if (!_head || _head->_prev == _head) return _T();

	const _T val = _head->_prev->_val;
	Node* node = _head->_prev;

	erase_node(&node);

	_ptr_list[_n_used] = NULL;
	--_n_used;

	return val;
}

template <class _T>
size_t Vector<_T>::size() const {
	return _n_used;
}

template <class _T>
void Vector<_T>::resize(size_t size) {
	++_count_create;

	clear();

	_ptr_len = ADDR_ALLOC + (size / ADDR_ALLOC) * ADDR_ALLOC;
	_n_used = size;

	_ptr_list = create_ptr_list(_ptr_len);
	_head = create_head();
	
	for (size_t i = 0; i < size; ++i) {
		Node* node = new Node;

		insert_node(_head->_prev, node);
		_ptr_list[i] = node;
	}

	id = linker.create();

	--_count_create;
}

template <class _T>
typename Vector<_T>::Iterator Vector<_T>::begin() const {
	return Iterator(_head->_next);
}

template <class _T>
typename Vector<_T>::Iterator Vector<_T>::end() const {
	return Iterator(_head);
}

/**********************************************/
/*                                            */
/*                     List                   */
/*                                            */
/**********************************************/

template <class _T>
class List {
public:
	Vector<List> _list;

	_T _val;

	List();
	List(const _T& val);
	List(const std::initializer_list<_T>& list);
	List(const std::initializer_list<List>& list);
	List(const List& p);

	List& operator=(const List& p);
	List& operator[](int index);

	void clear();
	void push_back(const List& p);
	size_t size() const;

	void resize(size_t size);

	class Iterator {
	protected:
		bool _switch;
		const List* _this;

	public:
		typename Vector<List>::Iterator _iter;

		Iterator(const List* curr_this, typename Vector<List>::Iterator current_iter, bool is_scalar);

		void operator++();
		bool operator!=(typename const Iterator& p) const;
		const List<_T>& operator*() const;
	};

	typename Iterator begin() const;
	typename Iterator end() const;
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
void List<_T>::clear() {
	_list.clear();
	_val = _T();
}

template <class _T>
void List<_T>::push_back(const List<_T>& p) {
	_list.push_back(p);
}

template <class _T>
size_t List<_T>::size() const {
	return _list.size();
}

template <class _T>
void List<_T>::resize(size_t size) {
	if (size > 1) _list.resize(size);
}

template <class _T>
List<_T>::Iterator::Iterator(const List<_T>* curr_this, typename Vector<List<_T>>::Iterator current_iter, bool is_scalar) :
	_this(curr_this),
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
bool List<_T>::Iterator::operator!=(typename const Iterator& p) const {
	if (_switch) return true;
	else return _iter != p._iter;
}

template <class _T>
const List<_T>& List<_T>::Iterator::operator*() const {
	if (_switch) return *_this;
	else return *_iter;
}

template <class _T>
typename List<_T>::Iterator List<_T>::begin() const {
	if (_list.size() > 0) return Iterator(this, _list.begin(), false);
	else return Iterator(this, _list.begin(), true);
}

template <class _T>
typename List<_T>::Iterator List<_T>::end() const {
	if (_list.size() > 0) return Iterator(this, _list.end(), false);
	else return Iterator(this, _list.end(), true);
}


/**********************************************/
/*                                            */
/*                   nn_shape                 */
/*                                            */
/**********************************************/

typedef List<int> nn_shape;

const char* put_shape(const nn_shape& tensor);