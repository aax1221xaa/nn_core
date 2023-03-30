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


#if false
/**********************************************/
/*                                            */
/*              _DoubleListNode               */
/*                                            */
/**********************************************/

template <class _T>
struct _ListNode {
	_ListNode* v_prev;
	_ListNode* v_next;
	
	_ListNode* h_prev;
	_ListNode* h_next;

	_T* object;
};

/**********************************************/
/*                                            */
/*                    List                    */
/*                                            */
/**********************************************/

template <class _T>
class List : public NN_Shared_Ptr {
protected:
	_ListNode<_T>* _head;

	_ListNode<_T>* _create_head();
	void _clear();
	vector<_ListNode<_T>*> find_heads();
	void v_insert(_T* _object, _ListNode<_T>* prev_node);
	void h_insert(_T* _object, _ListNode<_T>* prev_node);
	void v_erase(_ListNode<_T>** node);
	void h_erase(_ListNode<_T>** node);
	_ListNode<_T>* copy();

public:
	List();

	List(const _T val);
	List(const initializer_list<List> list);
	List(const initializer_list<_T> list);
	List(const List& p);
	~List();

	List& append(const List& list);
	List& extend(const List& list);

	List& operator=(const List& p);
	List operator[](const int index);
	void set(const _T& val);
	const _T& get();

	/*
	class Iterator {
	protected:
		_ListNode* m_head;
		_ListNode* m_current;

	public:
		Iterator(_ListNode* head, _ListNode* current);
		Iterator(typename const Iterator& p);

		typename const Iterator& operator++();
		typename const Iterator& operator--();
		bool operator!=(typename const Iterator& p) const;
		bool operator==(typename const Iterator& p) const;
		_T& operator*() const;
	};
	*/
};

/*
template <class _T>
List<_T>::Iterator::Iterator(_ListNode* head, _ListNode* current) {
	m_head = head;
	m_current = current;
}

template <class _T>
List<_T>::Iterator::Iterator(typename const Iterator& p) {
	m_head = p.m_head;
	m_current = p.m_current;
}

template <class _T>
typename const List<_T>::Iterator& List<_T>::Iterator::operator++() {
	m_current = m_current->_next;

	if (m_current == m_head) ErrorExcept("List::Iterator::operator++] The next node no longer exist.");

	return *this;
}

template <class _T>
typename const List<_T>::Iterator& List<_T>::Iterator::operator--() {
	m_current = m_current->_prev;

	if (m_current == m_head) ErrorExcept("List::Iterator::operator--] The prev node no longer exist.");

	return *this;
}

template <class _T>
bool List<_T>::Iterator::operator!=(typename const Iterator& p) const {
	return m_current != p.m_current;
}

template <class _T>
bool List<_T>::Iterator::operator==(typename const Iterator& p) const {
	return m_current == p.m_current;
}

template <class _T>
_T& List<_T>::Iterator::operator*() const {
	if (m_current->_is_list) ErrorExcept("[List::Iterator::operator*] list is not a value of current type");

	return *((_T*)m_current->_object);
}
*/

template <class _T>
_ListNode<_T>* List<_T>::_create_head() {
	_ListNode<_T>* head = new _ListNode<_T>;

	head->v_prev = head->v_next = head;
	head->h_prev = head->h_next = head;
	head->object = NULL;

	return head;
}

template <class _T>
void List<_T>::_clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			vector<_ListNode<_T>*> heads = find_heads();

			for (_ListNode<_T>* head : heads) {
				_ListNode<_T>* current = head->v_next;

				while (current != head) {
					if (current->h_next != current) {
						_ListNode<_T>* tmp = current->v_next;
						v_erase(&current);
						current = tmp;
					}
				}

				current = head->h_next;

				while (current != head) {
					if (current->v_next != current) {
						_ListNode<_T>* tmp = current->h_next;
						h_erase(&current);
						current = tmp;
					}
				}
			}

			for (_ListNode<_T>* head : heads) delete head;

			linker.Erase(id);
			id = NULL;
		}
	}
}

template <class _T>
vector<_ListNode<_T>*> List<_T>::find_heads() {
	vector<_ListNode<_T>*> heads;
	vector<_ListNode<_T>*> orders;

	heads.push_back(_head);
	orders.push_back(_head);

	while (!orders.empty()) {
		_ListNode<_T>* head = orders.front();
		_ListNode<_T>* current = head->v_next;

		while (current != head) {
			if (current->h_next != current) {
				heads.push_back(current);
				orders.push_back(current);
			}
			current = current->v_next;
		}

		current = head->h_next;

		while (current != head) {
			if (current->v_next != current) {
				heads.push_back(current);
				orders.push_back(current);
			}
			current = current->h_next;
		}

		orders.erase(orders.begin());
	}

	return heads;
}

template <class _T>
void List<_T>::v_insert(_T* _object, _ListNode<_T>* prev_node) {
	_ListNode<_T>* current = new _ListNode<_T>;

	_ListNode<_T>* before = _head->v_prev;
	_ListNode<_T>* after = _head;

	before->v_next = current;
	after->v_prev = current;
	current->v_next = after;
	current->v_prev = before;

	current->object = _object;
}

template <class _T>
void List<_T>::h_insert(_T* _object, _ListNode<_T>* prev_node) {
	_ListNode<_T>* current = new _ListNode<_T>;

	_ListNode<_T>* before = _head->h_prev;
	_ListNode<_T>* after = _head;

	before->h_next = current;
	after->h_prev = current;
	current->h_next = after;
	current->h_prev = before;

	current->object = _object;
}

template <class _T>
void List<_T>::v_erase(_ListNode<_T>** node) {
	_ListNode<_T>* v_before = (*node)->v_prev;
	_ListNode<_T>* v_after = (*node)->v_next;

	v_before->v_next = v_after;
	v_after->v_prev = v_before;

	delete (*node)->object;
	delete *node;

	*node = NULL;
}

template <class _T>
void List<_T>::h_erase(_ListNode<_T>** node) {
	_ListNode<_T>* h_before = (*node)->h_prev;
	_ListNode<_T>* h_after = (*node)->h_next;

	h_before->h_next = h_after;
	h_after->h_prev = h_before;

	delete (*node)->object;
	delete *node;

	*node = NULL;
}

template <class _T>
_ListNode<_T>* List<_T>::copy() {
	_ListNode<_T>* m_head = _create_head();
	_ListNode<_T>* m_current = m_head;
	vector<_ListNode<_T>*> orders;

	orders.push_back(_head);

	while (!orders.empty()) {
		_ListNode<_T>* head = orders.front();
		_ListNode<_T>* current = head->h_next;
		
		while (current != head) {
			if (current->v_next != current) {
				orders.push_back(current);
				v_insert(NULL, m_current);
			}
		}
	}
}

template <class _T>
List<_T>::List(const _T val) {
	_head = _create_head();

	h_insert(new _T(val), _head);
	id = linker.Create();
}

template <class _T>
List<_T>::List(const initializer_list<_T> list) {
	_head = _create_head();

	for (const _T& p : list) {
		h_insert(new _T(p), _head);
	}
	id = linker.Create();
}

template <class _T>
List<_T>::List() {
	_head = _create_head();
	id = linker.Create();
}

template <class _T>
List<_T>::List(const initializer_list<List> list) {
	_ListNode<_T>* current = new _ListNode<_T>;
	_head = current;

	for (const List<_T>& p : list) {
		current->h_next = current->h_prev = current;
		current->v_next = _head;
		current->v_prev = _head->h_prev;
		heads.back()->object = p._head->object;
	}
	_head = heads.front();

	id = linker.Create();
}

template <class _T>
List<_T>::List(const List& p) {
	_head = p._head;
	id = p.id;
	++id->ref_cnt;
}

template <class _T>
List<_T>::~List() {
	_clear();
}

template <class _T>
List<_T>& List<_T>::append(const List& list) {
	_insert((void*)&list, true, true);

	return *this;
}

template <class _T>
List<_T>& List<_T>::extend(const List& list) {
	_ListNode* current = list._head->_next;

	while (current != list._head) {
		_insert(current->_object, true, current->_is_list);
		current = current->_next;
	}

	return *this;
}

template <class _T>
List<_T>& List<_T>::operator=(const List& p) {
	if (&p == this) return *this;

	_clear();
	_head = p._head;
	id = p.id;

	++id->ref_cnt;

	return *this;
}

template <class _T>
List<_T> List<_T>::operator[](const int index) {
	_ListNode* node = _head->_next;

	for (int i = 0; i < index; ++i) {
		if (node == _head) {
			ErrorExcept("[List::operator[]] invalid index.");
		}
		node = node->_next;
	}

	return node->_is_list ? List<_T>(*((List<_T>*)node->_object)) : List<_T>(*((_T*)node->_object));
}

template <class _T>
void List<_T>::set(const _T& val) {
	if (_head->_next->_is_list) ErrorExcept("[List::set] list can't set value.");

	*((_T*)_head->_next->_object) = val;
}

template <class _T>
const _T& List<_T>::get() {
	if (_head->_next->_is_list) ErrorExcept("[List::get] list can't get value.");

	return *((_T*)_head->_next->_object);
}

/**********************************************/
/*                                            */
/*                  print_list                */
/*                                            */
/**********************************************/

template <class _T>
void print_list(const List<_T>& list) {

}

#else

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


#endif
