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
/*                  ListNode                  */
/*                                            */
/**********************************************/

template <class _T>
class ListNode {
public:
	ListNode* _prev;
	ListNode* _next;
	ListNode* _nodes;

	_T _val;

	const char* _label;

	ListNode(const char* label = "ListNode");
	virtual ~ListNode();
	
	virtual cuint get_capacity();
	virtual void put(ostream& os) const;

	static void insert_link(ListNode* current, ListNode* prev_node);
	static void clear_link(ListNode* current);
};

template <class _T>
ListNode<_T>::ListNode(const char* label) :
	_prev(NULL),
	_next(NULL),
	_nodes(NULL),
	_label(label)
{
}

template <class _T>
ListNode<_T>::~ListNode() {
	delete _nodes;
	_nodes = NULL;
}

template <class _T>
cuint ListNode<_T>::get_capacity() {
	return 0;
}

template <class _T>
void ListNode<_T>::put(ostream& os) const {
	os << _val << ", ";
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
void ListNode<_T>::clear_link(ListNode* current) {
	ListNode<_T>* before = current->_prev;
	ListNode<_T>* after = current->_next;

	before->_next = after;
	after->_prev = before;

	current->_next = NULL;
	current->_prev = NULL;
}

/**********************************************/
/*                                            */
/*                ListNodeBlock               */
/*                                            */
/**********************************************/

template <class _T>
class ListNodeBlock : public ListNode<_T> {
public:
	uint _capacity;

	ListNodeBlock(cuint capacity = 32);
	~ListNodeBlock();

	cuint get_capacity();
	void put(ostream& os) const;

	static ListNode<_T>* find_free_node(ListNode<_T>* block);
};

template <class _T>
ListNodeBlock<_T>::ListNodeBlock(cuint capacity) :
	ListNode<_T>("ListNodeBlock")
{
	_capacity = capacity;

	if (capacity > 0) {
		this->_nodes = new ListNode<_T>[capacity];
		this->_nodes[0]._next = this->_nodes;
		this->_nodes[0]._next = this->_nodes;
	}
}

template <class _T>
ListNodeBlock<_T>::~ListNodeBlock() {
	delete[] this->_nodes;
	this->_nodes = NULL;
}

template <class _T>
cuint ListNodeBlock<_T>::get_capacity() {
	return _capacity;
}

template <class _T>
void ListNodeBlock<_T>::put(ostream& os) const {
	ListNode<_T>* head_node = this->_nodes;
	ListNode<_T>* current_node = head_node;

	do {
		if (current_node->_nodes) current_node->_nodes->put(os);
		else current_node->put(os);
		current_node = current_node->_next;

	} while (current_node != head_node);
}

template <class _T>
ListNode<_T>* ListNodeBlock<_T>::find_free_node(ListNode<_T>* block) {
	ListNode<_T>* free_node = NULL;
	ListNode<_T>* nodes = block->_nodes;
	cuint capacity = block->get_capacity();

	for (uint i = 0; i < capacity; ++i) {
		if (nodes[i]._next == NULL) {
			free_node = nodes[i]._next;
			break;
		}
	}

	return free_node;
}

/**********************************************/
/*                                            */
/*                     List                   */
/*                                            */
/**********************************************/

template <class _T>
class List : public ListNode<_T>, public NN_Shared_Ptr {
protected:
	static ListNodeBlock<_T>* create_nodes(cuint capacity);

public:
	List();
	List(const _T& val);
	List(const initializer_list<_T>& list, cuint capacity = 4);
	List(const initializer_list<List>& list, cuint capacity = 4);
	List(const List& p);
	~List();

	List& operator=(const List& p);

	void put(ostream& os) const;
};

template <class _T>
ListNodeBlock<_T>* List<_T>::create_nodes(cuint capacity) {
	ListNodeBlock<_T>* nodes = new ListNodeBlock<_T>(capacity);

	nodes->_prev = nodes;
	nodes->_next = nodes;

	return nodes;
}

template <class _T>
List<_T>::List() :
	ListNode<_T>("List")
{
	id = NULL;
}

template <class _T>
List<_T>::List(const _T& val) :
	ListNode<_T>("List")
{
	this->_val = val;

	id = NULL;
}

template <class _T>
List<_T>::List(const initializer_list<_T>& list, cuint capacity) :
	ListNode<_T>("List")
{
	this->_nodes = create_nodes(capacity);

	uint i = 0;
	ListNode<_T>* current_block = NULL;
	ListNode<_T>* prev_block = this->_nodes;
	ListNode<_T>* prev_node = NULL;

	for (const _T& val : list) {
		if (i == 0) {
			current_block = current_block == NULL ? this->_nodes : new ListNodeBlock<_T>(capacity);
			ListNode<_T>::insert_link(current_block, prev_block);
			prev_block = current_block;
			prev_node = current_block->_nodes;
		}
		
		ListNode<_T>& current_node = current_block->_nodes[i];
		
		current_node._val = val;
		ListNode<_T>::insert_link(&current_node, prev_node);
		prev_node = &current_node;

		i = (i + 1) % capacity;
	}

	id = linker.Create();
}

template <class _T>
List<_T>::List(const initializer_list<List>& list, cuint capacity) :
	ListNode<_T>("List")
{
	this->_nodes = create_nodes(capacity);

	uint i = 0;
	ListNode<_T>* new_block = NULL;
	ListNode<_T>* prev_block = this->_nodes;
	ListNode<_T>* prev_node = NULL;

	for (const List<_T>& p_list : list) {
		if (i == 0) {
			new_block = new_block == NULL ? this->_nodes : new ListNodeBlock<_T>(capacity);
			ListNode<_T>::insert_link(new_block, prev_block);
			prev_block = new_block;
			prev_node = new_block->_nodes;
		}

		ListNode<_T>& new_node = new_block->_nodes[i];

		if (p_list._nodes == NULL) {
			new_node._val = p_list._val;
		}
		else {
		new_node._nodes = new List(p_list);
		}

		ListNode<_T>::insert_link(&new_node, prev_node);
		prev_node = &new_node;

		i = (i + 1) % capacity;
	}

	id = linker.Create();
}

template <class _T>
List<_T>::List(const List& p) :
	ListNode<_T>("List")
{
	this->_val = p._val;
	this->_nodes = p._nodes;

	id = p.id;

	if (id) ++id->ref_cnt;
}

template <class _T>
List<_T>::~List() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			ListNode<_T>* current = this->_nodes->_next;
			ListNode<_T>* tmp = NULL;

			while (current != this->_nodes) {
				tmp = current->_next;
				delete current;
				current = tmp;
			}
			delete this->_nodes;

			linker.Erase(id);
		}
	}

	this->_nodes = NULL;
	id = NULL;
}

template <class _T>
List<_T>& List<_T>::operator=(const List<_T>& p) {
	if (this == &p) return *this;

	this->_val = p._val;
	this->_nodes = p._nodes;
	
	this->id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <class _T>
void List<_T>::put(ostream& os) const {
	ListNode<_T>* head = this->_nodes;
	ListNode<_T>* current = head;

	os << '[';

	if (head == NULL) {
		os << this->_val;
	}
	else {
		do {
			current->put(os);
			current = current->_next;
		} while (current != head);
	}

	os << "], ";
}

template <class _T>
ostream& operator<<(ostream& os, const List<_T>& list) {
	list.put(os);

	return os;
}
