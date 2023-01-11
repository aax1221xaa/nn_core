#pragma once
#include "nn_base_layer.h"


template <class _T>
struct NN_Link_Param {
	_T* link;
	NN_Tensor* output;
	NN_Tensor* d_input;
	NN_Shape* out_size;
};

template <class _T>
struct NN_List_Node {
	_T* ptr;
	
	NN_List_Node* h_prev;
	NN_List_Node* h_next;
	NN_List_Node* v_prev;
	NN_List_Node* v_next;
};

template <class _T>
class NN_List {
public:
	NN_List_Node<_T>* head;

	class Iterator {
	public:
		NN_List_Node<_T>* p_node;

		Iterator(int* _shape, int _index);
		Iterator(const Iterator& p);

		const Iterator& operator++();
		const Iterator& operator--();
		bool operator!=(const Iterator& p) const;
		bool operator==(const Iterator& p) const;
		int& operator*() const;
	};

	NN_List();
	NN_List(const NN_List<_T>& p);
	NN_List(const initializer_list<NN_List<_T>>& list);
	~NN_List();

	void push_back(const NN_List<_T>& p);
	void push_back(const initializer_list<NN_List<_T>>& list);
	void clear();

	NN_List<_T> operator[](int index);
	const Iterator begin() const;
	const Iterator end() const;
};

template <class _T>
NN_List<_T>::NN_List() {
	head = new NN_List_Node<_T>;

	head->h_prev = head;
	head->h_next = head;
	head->v_prev = head;
	head->v_next = head;
}

template <class _T>
NN_List<_T>::NN_List(const NN_List<_T>& p) {

}

template <class _T>
NN_List<_T>::NN_List(const initializer_list<NN_List>& list) {

}

template <class _T>
NN_List<_T>::~NN_List() {
	NN_Link_Param<_T>* p_current = head->next_node;
	NN_Link_Param<_T>* tmp = NULL;

	while (p_current != head) {
		tmp = p_current->next_node;
		delete p_current;
		p_current = tmp;
	}
	delete head;
}

template <class _T>
void NN_List<_T>::push_back(const NN_List<_T>& p) {
	NN_Link_Param<_T>* p_current = new NN_Link_Param<_T>;

	NN_Link_Param<_T>* before = head->prev_node;
	NN_Link_Param<_T>* after = head;

	before->next_node = p_current;
	after->prev_node = p_current;
	p_current->prev_node = before;
	p_current->next_node = after;

	++len;
}

template <class _T>
void NN_List<_T>::push_back(const initializer_list<NN_List<_T>>& list) {

}

template <class _T>
void NN_List<_T>::clear() {

}

template <class _T>
NN_Link_Param<_T>& NN_List<_T>::operator[](int index) {

}

template <class _T>
const NN_List<_T>::Iterator NN_List<_T>::begin() const {

}

template <class _T>
const NN_List<_T>::Iterator NN_List<_T>::end() const {

}




class NN_Link {
public:
	bool is_selected;
	bool trainable;

	vector<NN_Link*> prev_link;
	vector<NN_Link*> next_link;

	NN_Link* parent;
	vector<NN_Link*> child;

	vector<NN_Tensor*> input;
	NN_Tensor output;

	NN_Tensor d_input;
	vector<NN_Tensor*> d_output;

	vector<NN_Shape*> in_shape;
	NN_Shape out_shape;

	NN_Layer* op_layer;

	NN_Link();
	virtual ~NN_Link();

	virtual NN_List<NN_Link> operator()(const NN_List<NN_Link> m_prev_link);
	void operator()(NN_Link* m_prev_link);

	virtual NN_Link* create_child_link();
};

typedef NN_List<NN_Link> NN;