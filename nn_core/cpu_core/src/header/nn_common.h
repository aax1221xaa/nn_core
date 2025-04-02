#pragma once
#include <memory>
#include <vector>
#include "Exception.h"


typedef float nn_type;
typedef unsigned int uint;
typedef const unsigned int cuint;


std::vector<int> random_choice(int min, int max, int amounts, bool replace);

#if 0
/**********************************************/
/*                                            */
/*                   NN_Vector				  */
/*                                            */
/**********************************************/

template <class _T>
class NN_Vector {
protected:
	class ConteinerNode {
	public:
		ConteinerNode* _prev;
		ConteinerNode* _next;

		size_t _capacity;
		size_t _len;
		_T* _data;

		class Iterator {
		public:
			_T* m_data;
			size_t m_index;

			Iterator(_T* data, size_t index) : m_data(data), m_index(index) {}
			Iterator(const typename Iterator& p) : m_data(p.m_data), m_index(p.m_index) {}

			void operator++() { ++m_index; }
			bool operator!=(const typename Iterator& p) { return m_index != p.m_index; }
			bool operator!=(const typename Iterator& p) const { return m_index != p.m_index; }
			_T& operator*() { return m_data[m_index]; }
			const _T& operator*() const { return m_data[m_index]; }
		};

		ConteinerNode();
		ConteinerNode(const _T* data, size_t len);
		ConteinerNode(size_t len, const _T& val);
		ConteinerNode(size_t capacity);
		ConteinerNode(const ConteinerNode& p);
		~ConteinerNode();

		_T& operator[](size_t index);
		const _T& operator[](size_t index) const;

		typename Iterator begin();
		typename Iterator end();
		typename Iterator begin() const;
		typename Iterator end() const;
	};

	std::shared_ptr<ConteinerNode> _head;
	size_t _curr_index;
	size_t _capacity;

	static ConteinerNode* create_head();
	static void insert_node(ConteinerNode* prev_node, ConteinerNode* current_node);
	static void remove_node(ConteinerNode* current_node);
	static ConteinerNode* copy_head(const ConteinerNode* head);
	static void del_func(ConteinerNode* head);

public:
	class Iterator {
	public:
		ConteinerNode* m_head;

		Iterator() : m_head(NULL) {}
		Iterator(const typename Iterator& p) : m_head(p.m_head) {}

		void operator++() { m_head = m_head->_next; }
		bool operator!=(const typename Iterator& p) { return m_head != p.m_head; }
		bool operator!=(const typename Iterator& p) const { return m_head != p.m_head; }
		ConteinerNode& operator*() { return *m_head; }
		const ConteinerNode& operator*() const { return *m_head; }
	};

	NN_Vector();
	NN_Vector(size_t len, const _T& val);
	NN_Vector(const std::initializer_list<_T>& list);
	NN_Vector(const std::vector<_T>& vect);
	NN_Vector(const _T* ptr, size_t len);
	NN_Vector(const NN_Vector& p);
	NN_Vector(NN_Vector&& p);
	~NN_Vector();

	NN_Vector& operator=(const NN_Vector& p);
	NN_Vector& operator=(NN_Vector&& p);

	typename Iterator begin();
	typename Iterator end();
	typename Iterator begin() const;
	typename Iterator end() const;
};

template <class _T>
NN_Vector<_T>::ConteinerNode::ConteinerNode() :
	_prev(NULL),
	_next(NULL),
	_capacity(0),
	_len(0),
	_data(NULL)
{

}

template <class _T>
NN_Vector<_T>::ConteinerNode::ConteinerNode(const _T* data, size_t len) :
	_prev(NULL),
	_next(NULL),
	_capacity(len < 10 ? 10 : len),
	_len(len),
	_data(new _T[_capacity])
{
	for (size_t i = 0; i < len; ++i) _data[i] = data[i];
}

template <class _T>
NN_Vector<_T>::ConteinerNode::ConteinerNode(size_t len, const _T& val) :
	_prev(NULL),
	_next(NULL),
	_capacity(len < 10 ? 10 : len),
	_len(len),
	_data(new _T[_capacity])
{
	for (size_t i = 0; i < len; ++i) _data[i] = val;
}

template <class _T>
NN_Vector<_T>::ConteinerNode::ConteinerNode(size_t len) :
	_prev(NULL),
	_next(NULL),
	_capacity(len < 10 ? 10 : len),
	_len(len),
	_data(new _T[_capacity])
{

}

template <class _T>
NN_Vector<_T>::ConteinerNode::ConteinerNode(const ConteinerNode& p) :
	_prev(NULL),
	_next(NULL),
	_capacity(p._capacity),
	_len(p._len),
	_data(new _T[p._capacity])
{
	for (size_t i = 0; i < _len; ++i) _data[i] = p._data[i];
}

template <class _T>
NN_Vector<_T>::ConteinerNode::~ConteinerNode() {
	delete[] _data;
}

template <class _T>
_T& NN_Vector<_T>::ConteinerNode::operator[](size_t index) {
	if (_len <= index) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::operator[]] index is over the length."
		);
	}

	return _data[index];
}

template <class _T>
const _T& NN_Vector<_T>::ConteinerNode::operator[](size_t index) const {
	if (_len <= index) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::operator[]] index is over the length."
		);
	}

	return _data[index];
}

template <class _T>
typename NN_Vector<_T>::ConteinerNode::Iterator NN_Vector<_T>::ConteinerNode::begin() {
	return Iterator(_data, 0);
}

template <class _T>
typename NN_Vector<_T>::ConteinerNode::Iterator NN_Vector<_T>::ConteinerNode::end() {
	return Iterator(_data, _len);
}

template <class _T>
typename NN_Vector<_T>::ConteinerNode::Iterator NN_Vector<_T>::ConteinerNode::begin() const {
	return Iterator(_data, 0);
}

template <class _T>
typename NN_Vector<_T>::ConteinerNode::Iterator NN_Vector<_T>::ConteinerNode::end() const {
	return Iterator(_data, _len);
}

template <class _T>
NN_Vector<_T>::ConteinerNode* NN_Vector<_T>::create_head() {
	ConteinerNode* head = new ConteinerNode();

	head->_prev = head;
	head->_next = head;

	return head;
}

template <class _T>
void NN_Vector<_T>::insert_node(ConteinerNode* prev_node, ConteinerNode* current_node) {
	if (prev_node == NULL) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::insert_node] prev_node is null pointer."
		);
	}
	else if (current_node == NULL) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::insert_node] current_node is null pointer."
		);
	}
	else if (prev_node->_prev == NULL) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::insert_node] prev_node->_prev is null pointer."
		);
	}
	else if (prev_node->_next == NULL) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::insert_node] prev_node->_next is null pointer."
		);
	}

	ConteinerNode* before = prev_node->_prev;
	ConteinerNode* after = prev_node->_next;

	before->_next = current_node;
	after->_prev = current_node;
	current_node->_prev = before;
	current_node->_next = after;
}

template <class _T>
void NN_Vector<_T>::remove_node(ConteinerNode* current_node) {
	if (current_node == NULL) {
		ErrorExcept(
			"[NN_Vector<_T>::ConteinerNode::remove_node] current_node is null pointer."
		);
	}

	ConteinerNode* before = current_node->_prev;
	ConteinerNode* after = current_node->_next;

	before->_next = after;
	after->_prev = before;

	delete current_node;
}

template <class _T>
NN_Vector<_T>::ConteinerNode* NN_Vector<_T>::copy_head(const ConteinerNode* head) {
	if (head == NULL) {
		ErrorExcept(
			"[NN_Vector<_T>::copy_head] head is null pointer."
		);
	}

	ConteinerNode* m_head = create_head();
	ConteinerNode* current_node = head->_next;
	
	while (current_node != head) {
		ConteinerNode* node = new ConteinerNode(*current_node);
		
		insert_node(m_head->_prev, node);
		current_node = current_node->_next;
	}
}

template <class _T>
void NN_Vector<_T>::del_func(ConteinerNode* head) {
	ConteinerNode* tmp = head->_next;

	while (tmp != head) {
		tmp = tmp->_next;
		delete tmp->_prev;
	}

	delete head;
	head = NULL;

}

template <class _T>
NN_Vector<_T>::NN_Vector() :
	_curr_index(0),
	_capacity(10)
{
	_head = NULL;
}

template <class _T>
NN_Vector<_T>::NN_Vector(size_t len, const _T& val) :
	_curr_index(0),
	_capacity(10)
{
	_head = std::shared_ptr<ConteinerNode>(create_head(), del_func);

	ConteinerNode* node = new ConteinerNode(len, val);

	try {
		insert_node(_head.get(), node);
	}
	catch (const NN_Exception& e) {
		_head = NULL;
		delete node;

		throw e;
	}
}

template <class _T>
NN_Vector<_T>::NN_Vector(const std::initializer_list<_T>& list) :
	_curr_index(0),
	_capacity(10)
{
	_head = std::shared_ptr<ConteinerNode>(create_head(), del_func);

	try {
		ConteinerNode* node = new ConteinerNode(list.size());
		typename ConteinerNode::Iterator node_iter = node->begin();

		for (const _T& elem : list) {
			*node_iter = elem;
			++node_iter;
		}

		insert_node(_head.get(), node);
	}
	catch (const NN_Exception& e) {
		_head = NULL;

		throw e;
	}
}

template <class _T>
NN_Vector<_T>::NN_Vector(const std::vector<_T>& vect) :
	_curr_index(0),
	_capacity(10)
{
	_head = std::shared_ptr<ConteinerNode>(create_head(), del_func);

	try {
		ConteinerNode* node = new ConteinerNode(vect.size());
		typename ConteinerNode::Iterator node_iter = node->begin();

		for (const _T& elem : vect) {
			*node_iter = elem;
			++node_iter;
		}

		insert_node(_head.get(), node);
	}
	catch (const NN_Exception& e) {
		_head = NULL;

		throw e;
	}
}

template <class _T>
NN_Vector<_T>::NN_Vector(const _T* ptr, size_t len) :
	_curr_index(0),
	_capacity(10)
{
	_head = std::shared_ptr<ConteinerNode>(create_head(), del_func);

	try {
		ConteinerNode* node = new ConteinerNode(ptr, len);
		typename ConteinerNode::Iterator node_iter = node->begin();

		for (size_t i = 0; i < len; ++i) {
			*node_iter = ptr[i];
			++node_iter;
		}

		insert_node(_head.get(), node);
	}
	catch (const NN_Exception& e) {
		_head = NULL;

		throw e;
	}
}

template <class _T>
NN_Vector<_T>::NN_Vector(const NN_Vector& p) :
	_curr_index(p._curr_index),
	_capacity(p._capacity)
{
	try {
		_head = std::shared_ptr<ConteinerNode>(copy_head(p._head.get()), del_func);
	}
	catch (const NN_Exception& e) {
		_head = NULL;

		throw e;
	}
}

template <class _T>
NN_Vector<_T>::NN_Vector(NN_Vector&& p) :
	_curr_index(p._curr_index),
	_capacity(p._capacity),
	_head(p._head)
{
	p._head = NULL;
}

template <class _T>
NN_Vector<_T>::~NN_Vector() {

}

template <class _T>
NN_Vector<_T>& NN_Vector<_T>::operator=(const NN_Vector& p) {
	if (this == &p) return *this;

	_curr_index = p._curr_index;
	_capacity = p._capacity;
	_head = std::shared_ptr<ConteinerNode>(copy_head(p._head.get()), del_func);

	return *this;
}

template <class _T>
NN_Vector<_T>& NN_Vector<_T>::operator=(NN_Vector&& p) {
	_curr_index = p._curr_index;
	_capacity = p._capacity;
	_head = p._head;

	p._head = NULL;
}

template <class _T>
typename NN_Vector<_T>::Iterator NN_Vector<_T>::begin() {

}

template <class _T>
typename NN_Vector<_T>::Iterator NN_Vector<_T>::end() {

}

template <class _T>
typename NN_Vector<_T>::Iterator NN_Vector<_T>::begin() const {

}

template <class _T>
typename NN_Vector<_T>::Iterator NN_Vector<_T>::end() const {

}

#endif