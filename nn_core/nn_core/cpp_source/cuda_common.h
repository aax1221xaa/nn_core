#pragma once
#include <vector>
#include <iostream>
#include "CudaCheck.h"
#include "ObjectID.h"


using namespace std;

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

#define STR_MAX			1024

#define BLOCK_SIZE				32
#define SQR_BLOCK_SIZE			BLOCK_SIZE * BLOCK_SIZE
#define CONST_ELEM_SIZE			16384
#define CONST_MEM_SIZE			65536

#define EPSILON					1e-8


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
/*                  ListNode                  */
/*                                            */
/**********************************************/

template <class _T>
class ListNode {
public:
	ListNode* _prev;
	ListNode* _next;
	ListNode* _head;

	_T _val;

	int _counts;

	class Iterator {
	protected:
		ListNode<_T>* _p_node;
		int _index;

	public:
		Iterator(ListNode<_T>* p_node, int index);
		Iterator(typename const Iterator& p);

		typename const Iterator& operator++();
		typename const Iterator& operator--();
		bool operator!=(typename const Iterator& p) const;
		bool operator==(typename const Iterator& p) const;
		ListNode<_T>& operator*() const;
	};

	ListNode();
	ListNode(const _T& val);
	virtual ~ListNode();

	static void insert_link(ListNode* current, ListNode* prev_node);
	static void remove_link(ListNode* current);

	typename const Iterator begin() const;
	typename const Iterator end() const;

	//virtual void print(ostream& os);
};

template <class _T>
ListNode<_T>::Iterator::Iterator(ListNode<_T>* p_node, int index) {
	_p_node = p_node;
	_index = index;
}

template <class _T>
ListNode<_T>::Iterator::Iterator(typename const Iterator& p) {
	_p_node = p._p_node;
	_index = p._index;
}

template <class _T>
typename const ListNode<_T>::Iterator& ListNode<_T>::Iterator::operator++() {
	_p_node = _p_node->_next;
	++_index;

	return *this;
}

template <class _T>
typename const ListNode<_T>::Iterator& ListNode<_T>::Iterator::operator--() {
	_p_node = _p_node->_prev;
	--_index;

	return *this;
}

template <class _T>
bool ListNode<_T>::Iterator::operator!=(typename const ListNode<_T>::Iterator& p) const {
	return _index != p._index;
}

template <class _T>
bool ListNode<_T>::Iterator::operator==(typename const ListNode<_T>::Iterator& p) const {
	return _index == p._index;
}

template <class _T>
ListNode<_T>& ListNode<_T>::Iterator::operator*() const {
	return *_p_node;
}

template <class _T>
ListNode<_T>::ListNode() :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_counts(0)
{
}

template <class _T>
ListNode<_T>::ListNode(const _T& val) :
	_prev(NULL),
	_next(NULL),
	_head(NULL),
	_counts(0),
	_val(val)
{
}

template <class _T>
ListNode<_T>::~ListNode() {
	delete _head;
	_head = NULL;
}

template <class _T>
void ListNode<_T>::insert_link(ListNode* current, ListNode* prev_node) {
	ListNode<_T>* before = prev_node;
	ListNode<_T>* after = prev_node->_next;

	before->_next = current;
	after->_prev = current;

	current->_next = after;
	current->_prev = before;
}

template <class _T>
void ListNode<_T>::remove_link(ListNode* current) {
	ListNode<_T>* before = current->_prev;
	ListNode<_T>* after = current->_next;

	before->_next = after;
	after->_prev = before;

	current->_next = NULL;
	current->_prev = NULL;
}

template <class _T>
typename const ListNode<_T>::Iterator ListNode<_T>::begin() const {
	return ListNode<_T>::Iterator(this->_head, 0);
}

template <class _T>
typename const ListNode<_T>::Iterator ListNode<_T>::end() const {
	return ListNode<_T>::Iterator(this->_head->_prev, _counts);
}

/*
template <class _T>
void ListNode<_T>::print(ostream& os) {
	os << _val << ", ";
}
*/

/**********************************************/
/*                                            */
/*                     List                   */
/*                                            */
/**********************************************/

template <class _T>
class List : public ListNode<_T>, public NN_Shared_Ptr {
protected:
	static ListNode<_T>* create_head();

public:
	List();
	List(const _T& val);
	List(const initializer_list<_T>& list);
	//List(const initializer_list<List>& list);
	List(initializer_list<List>&& list);
	List(const List& p);
	~List();

	void push_back(const _T& val);

	List& operator=(const List& p);
	_T& operator[](const int index);

	//void print(ostream& os);
};

template <class _T>
ListNode<_T>* List<_T>::create_head() {
	ListNode<_T>* head = new ListNode<_T>;

	head->_prev = head;
	head->_next = head;

	return head;
}

template <class _T>
List<_T>::List() {
	this->create_head = create_head();

	id = linker.Create();

	cout << "()" << endl;
}

template <class _T>
List<_T>::List(const _T& val) {
	this->_head = create_head();
	this->_head->_val = val;
	++this->_counts;

	id = linker.Create();

	cout << "const _T& val" << endl;
	//cout << *this;
}

template <class _T>
List<_T>::List(const initializer_list<_T>& list) {
	ListNode<_T>* current_node = NULL;
	this->_head = create_head();

	for (const _T& val : list) {
		current_node = current_node == NULL ? this->_head : new ListNode<_T>;
		
		current_node->_val = val;
		ListNode<_T>::insert_link(current_node, this->_head->_prev);

		++this->_counts;
	}

	id = linker.Create();

	cout << "const initializer_list<_T>&" << endl;
	//cout << *this;
}

/*
template <class _T>
List<_T>::List(const initializer_list<List>& list) {
	ListNode<_T>* current_node = NULL;
	this->_head = create_head();

	for (const List<_T>& p_list : list) {
		current_node = current_node == NULL ? this->_head : new ListNode<_T>;

		current_node->_head = new List<_T>(p_list);
		ListNode<_T>::insert_link(current_node, this->_head->_prev);

		++this->_counts;
	}

	id = linker.Create();

	cout << "const initializer_list<List>&" << endl;
	cout << *this;
}
*/

template <class _T>
List<_T>::List(initializer_list<List>&& list) {
	ListNode<_T>* current_node = NULL;
	this->_head = create_head();

	for (const List<_T>& p_list : list) {
		current_node = current_node == NULL ? this->_head : new ListNode<_T>;

		current_node->_head = new List<_T>(p_list);
		ListNode<_T>::insert_link(current_node, this->_head->_prev);

		++this->_counts;
	}

	id = linker.Create();

	cout << "initializer_list<List>&& list" << endl;
	//cout << *this;
}

template <class _T>
List<_T>::List(const List& p) {
	this->_val = p._val;
	this->_head = p._head;
	this->_counts = p._counts;

	id = p.id;

	if (id) ++id->ref_cnt;

	cout << "const List& p" << endl;
	//cout << *this;
}

template <class _T>
List<_T>::~List() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			ListNode<_T>* current = this->_head->_next;
			ListNode<_T>* tmp = NULL;

			while (current != this->_head) {
				tmp = current->_next;
				delete current;
				current = tmp;
			}
			delete this->_head;

			linker.Erase(id);
		}
	}

	this->_counts = 0;
	this->_head = NULL;
	id = NULL;

	cout << "destroy" << endl;
}

template <class _T>
void List<_T>::push_back(const _T& val) {
	ListNode<_T>::insert_link(new ListNode<_T>(val), this->_head->_prev);
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	this->_val = p._val;
	this->_head = p._head;
	this->_counts = p._counts;
	
	this->id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
_T& List<_T>::operator[](const int index) {
	ListNode<_T>* current = this->_head;

	for (int i = 0; i < index; ++i) current = current->_next;

	return current->_val;
}
/*
template <class _T>
void List<_T>::print(ostream& os) {
	ListNode<_T>* current = this->_head;

	os << '[';

	for (int i = 0; i < this->_counts; ++i) {
		if (current->_head) {
			current->_head->print(os);
		}
		else {
			current->print(os);
		}
		current = current->_next;
	}

	os << "], ";
}
*/

/*
template <class _T>
ostream& operator<<(ostream& os, List<_T>& p) {
	os << "List" << endl;
	p.print(os);
	os << endl;

	return os;
}
*/