#pragma once
#include <vector>
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
struct ListNode {
	int h_next;
	int h_prev;
	int v_next;
	int v_prev;

	_T* ptr;
};

/**********************************************/
/*                                            */
/*                ListNodeBlock               */
/*                                            */
/**********************************************/

template <class _T>
struct ListNodeBlock {
	ListNodeBlock<_T>* _next;
	ListNodeBlock<_T>* _prev;

	ListNode<_T>* _block;

	uint node_counts;
	uint free_nodes;
};

/**********************************************/
/*                                            */
/*                     List                   */
/*                                            */
/**********************************************/

template <class _T>
class List : public NN_Shared_Ptr {
protected:
	ListNodeBlock<_T>* head;

	void create_head();
	static void insert_block(ListNodeBlock<_T>* current_block, ListNodeBlock<_T>* prev_block);
	static void insert_h_node(const _T& val, const uint relative_index, ListNodeBlock<_T>* node_block);
	static uint find_free_h_node(ListNodeBlock<_T>* node_block);
	static uint find_free_v_node(ListNodeBlock<_T>* node_block);

public:
	List();
	List(const _T& val, const uint capacity = 32);
	List(const initializer_list<_T>& list);
	List(const initializer_list<List>& list);
	List(const List& p);
	~List();
};

template <class _T>
void List<_T>::create_head() {
	head = new ListNodeBlock<_T>;

	head->_next = head;
	head->_prev = head;

	head->_block = NULL;
	head->node_counts = 0;
	head->free_nodes = 0;
}

template <class _T>
void List<_T>::insert_block(ListNodeBlock<_T>* current_block, ListNodeBlock<_T>* prev_block) {
	ListNodeBlock<_T>* before = prev_block;
	ListNodeBlock<_T>* after = prev_block->_next;

	before->_next = current_block;
	after->_prev = current_block;
	current_block->_next = after;
	current_block->_prev = before;
}

template <class _T>
void List<_T>::insert_h_node(const _T& val, const uint relative_index, ListNodeBlock<_T>* node_block) {
	int h_next = 0;
	int h_prev = 0;

	ListNode<_T>* nodes = node_block->_block;
	ListNode<_T>* current_node = NULL;

	for (uint i = 0; i < relative_index; ++i) {
		current_node = &nodes[h_next];
		h_next = current_node->h_next;
		h_prev = current_node->h_prev;
	}

	uint free_index = find_free_h_node(node_block);
	nodes[free_index].h_next = 
}

template <class _T>
uint List<_T>::find_free_h_node(ListNodeBlock<_T>* node_block) {
	uint h_index = 0;

	for (uint i = 0; i < node_block->node_counts; ++i) {
		if (node_block->_block[i].h_next == -2) {
			h_index = i;
			break;
		}
	}

	return h_index;
}

template <class _T>
uint List<_T>::find_free_v_node(ListNodeBlock<_T>* node_block) {
	uint v_index = 0;

	for (uint i = 0; i < node_block->node_counts; ++i) {
		if (node_block->_block[i].v_next == -2) {
			v_index = i;
			break;
		}
	}

	return v_index;
}

template <class _T>
List<_T>::List() {

}

template <class _T>
List<_T>::List(const _T& val, const uint capacity) {

}

template <class _T>
List<_T>::List(const initializer_list<_T>& list) {

}

template <class _T>
List<_T>::List(const initializer_list<List>& list) {

}

template <class _T>
List<_T>::List(const List& p) {

}

template <class _T>
List<_T>::~List() {

}



#endif
