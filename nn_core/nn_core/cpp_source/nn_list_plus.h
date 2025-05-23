#pragma once
#include "cuda_common.h"
#include "Exception.h"

#define PUT_LIST


template <class _T>
class NN_List {
	enum class Attr {
		HEAD,
		NODE
	};

	class NN_ListBase {
	public:
		const Attr _attr;

		NN_ListBase(Attr attr);
		NN_ListBase(const NN_ListBase& p);
		virtual ~NN_ListBase();

		virtual _T* get_data();
		virtual const _T* get_data() const;

		virtual void set_clear_nodes();
		virtual std::vector<NN_ListBase*>* get_nodes();
		virtual const std::vector<NN_ListBase*>* get_nodes() const;
		virtual void set_nodes(NN_ListBase* p_node);
		virtual void set_reserve(size_t capacity);
		virtual size_t get_capacity();
		
		virtual const NN_ListBase& operator=(const NN_ListBase& p);
		virtual const NN_ListBase& operator=(NN_ListBase&& p);

		virtual void clear();

#ifdef PUT_LIST
		std::ostream& put_list(std::ostream& os) const;
#endif
	};

	class NN_ListNode : public NN_ListBase {
	public:
		_T _data;

		NN_ListNode();
		NN_ListNode(const NN_ListBase& p);
		NN_ListNode(NN_ListBase&& p);
		NN_ListNode(const _T& val);
		NN_ListNode(_T&& val);
		~NN_ListNode();

		_T* get_data();
		const _T* get_data() const;

		const NN_ListBase& operator=(const NN_ListBase& p);
		const NN_ListBase& operator=(NN_ListBase&& p);
	};

	class NN_ListHead : public NN_ListBase {
	public:
		std::vector<NN_ListBase*> _nodes;

		NN_ListHead();
		NN_ListHead(NN_ListBase* ptr);
		NN_ListHead(const NN_ListBase& p);
		NN_ListHead(NN_ListBase&& p);
		NN_ListHead(const std::initializer_list<_T>& list);
		~NN_ListHead();

		void set_clear_nodes();
		std::vector<NN_ListBase*>* get_nodes();
		const std::vector<NN_ListBase*>* get_nodes() const;
		void set_nodes(NN_ListBase* p_node);
		void set_reserve(size_t capacity);
		size_t get_capacity();

		const NN_ListBase& operator=(const NN_ListBase& p);
		const NN_ListBase& operator=(NN_ListBase&& p);

		void clear();
	};

	NN_ListBase* _node;
	bool _shared;

protected:
	NN_List(NN_ListBase* node);

public:
	class Iterator {
	public:
		bool _sw;
		NN_List _frame;
		typename std::vector<NN_ListBase*>::iterator _iter;

		Iterator(NN_List& list, typename std::vector<NN_ListBase*>::iterator iter);
		Iterator(const Iterator& p);

		void operator++();
		bool operator!=(const Iterator& p);
		NN_List& operator*();
	};

	class ConstIterator {
	public:
		bool _sw;
		const NN_List _frame;
		typename std::vector<NN_ListBase*>::const_iterator _iter;

		ConstIterator(const NN_List& list, typename std::vector<NN_ListBase*>::const_iterator iter);
		ConstIterator(const typename ConstIterator& p);

		void operator++();
		bool operator!=(const typename ConstIterator& p) const;
		const NN_List& operator*() const;
	};
	

	NN_List();
	NN_List(const _T& val);
	NN_List(_T&& val);
	NN_List(const std::initializer_list<_T>& list);
	NN_List(const std::initializer_list<NN_List>& list);
	NN_List(const NN_List& p);
	NN_List(NN_List&& p);
	~NN_List();

	const NN_ListBase* get_node() const;

	NN_List& operator=(const _T& val);
	NN_List& operator=(const NN_List& p);
	NN_List& operator=(NN_List&& p);
	NN_List& operator=(const std::initializer_list<_T>& list);
	
	NN_List& operator[](int index);
	NN_List& operator[](int index) const;

	void clear();

	void append(const _T& val);
	void append(_T&& val);
	void append(const std::initializer_list<_T>& list);
	void append(const NN_List& list);

	void extend(const _T& val);
	void extend(_T&& val);
	void extend(const std::initializer_list<_T>& list);
	void extend(const NN_List& list);

	_T& val();
	size_t size();
	void reserve(size_t size);
	void resize(size_t size);
	bool is_scalar();
	bool is_empty();

	_T& val() const;
	size_t size() const;
	bool is_scalar() const;
	bool is_empty() const;
};

/**********************************************/
/*                                            */
/*                 NN_ListBase                */
/*                                            */
/**********************************************/

template <class _T>
NN_List<_T>::NN_ListBase::NN_ListBase(Attr attr) :
	_attr(attr)
{

}

template <class _T>
NN_List<_T>::NN_ListBase::NN_ListBase(const NN_ListBase& p) :
	_attr(p._attr)
{

}

template <class _T>
NN_List<_T>::NN_ListBase::~NN_ListBase() {

}

template <class _T>
_T* NN_List<_T>::NN_ListBase::get_data() {
	return NULL;
}

template <class _T>
const _T* NN_List<_T>::NN_ListBase::get_data() const {
	return NULL;
}

template <class _T>
void NN_List<_T>::NN_ListBase::set_clear_nodes() {

}

template <class _T>
std::vector<typename NN_List<_T>::NN_ListBase*>* NN_List<_T>::NN_ListBase::get_nodes() {
	return NULL;
}

template <class _T>
const std::vector<typename NN_List<_T>::NN_ListBase*>* NN_List<_T>::NN_ListBase::get_nodes() const {
	return NULL;
}

template <class _T>
void NN_List<_T>::NN_ListBase::set_nodes(NN_ListBase* p_node) {
	
}

template <class _T>
void NN_List<_T>::NN_ListBase::set_reserve(size_t capacity) {

}

template <class _T>
size_t NN_List<_T>::NN_ListBase::get_capacity() {
	return 0;
}

template <class _T>
const typename NN_List<_T>::NN_ListBase& NN_List<_T>::NN_ListBase::operator=(const NN_ListBase& p) {
	return *this;
}

template <class _T>
const typename NN_List<_T>::NN_ListBase& NN_List<_T>::NN_ListBase::operator=(NN_ListBase&& p) {
	return *this;
}

template <class _T>
void NN_List<_T>::NN_ListBase::clear() {

}

#ifdef PUT_LIST

template <class _T>
std::ostream& NN_List<_T>::NN_ListBase::put_list(std::ostream& os) const {
	switch (_attr)
	{
	case Attr::NODE:
		if (get_data()) os << *get_data();
		break;
	case Attr::HEAD:
		os << '[';
		for (const NN_ListBase* p_node : *get_nodes()) {
			if (p_node) p_node->put_list(os);
			os << ", ";
		}
		os << ']';
		break;
	default:
		break;
	}

	return os;
}

#endif

/**********************************************/
/*                                            */
/*                 NN_ListNode                */
/*                                            */
/**********************************************/

template <class _T>
NN_List<_T>::NN_ListNode::NN_ListNode() :
	NN_ListBase(Attr::NODE)
{

}

template <class _T>
NN_List<_T>::NN_ListNode::NN_ListNode(const NN_ListBase& p) :
	NN_ListBase(Attr::NODE)
{
	if (p._attr != Attr::NODE) {
		ErrorExcept(
			"[NN_List<_T>::NN_ListNode::NN_ListNode] This argument is not a node."
		);
	}

	_data = *p.get_data();
}

template <class _T>
NN_List<_T>::NN_ListNode::NN_ListNode(NN_ListBase&& p) :
	NN_ListBase(Attr::NODE)
{
	if (p._attr != Attr::NODE) {
		ErrorExcept(
			"[NN_List<_T>::NN_ListNode::NN_ListNode] This argument is not a node."
		);
	}

	_data = *p.get_data();
}

template <class _T>
NN_List<_T>::NN_ListNode::NN_ListNode(const _T& val) :
	NN_ListBase(Attr::NODE),
	_data(val)
{

}

template <class _T>
NN_List<_T>::NN_ListNode::NN_ListNode(_T&& val) :
	NN_ListBase(Attr::NODE),
	_data(val)
{

}

template <class _T>
NN_List<_T>::NN_ListNode::~NN_ListNode() {

}

template <class _T>
_T* NN_List<_T>::NN_ListNode::get_data() {
	return &_data;
}

template <class _T>
const _T* NN_List<_T>::NN_ListNode::get_data() const {
	return &_data;
}

template <class _T>
const typename NN_List<_T>::NN_ListBase& NN_List<_T>::NN_ListNode::operator=(const NN_ListBase& p) {
	if (this == &p) return *this;

	if (p._attr != Attr::NODE) {
		ErrorExcept(
			"This argument is not a node."
		);
	}

	_data = *p.get_data();

	return *this;
}

template <class _T>
const typename NN_List<_T>::NN_ListBase& NN_List<_T>::NN_ListNode::operator=(NN_ListBase&& p) {
	if (this == &p) return *this;

	if (p._attr != Attr::NODE) {
		ErrorExcept(
			"This argument is not a node."
		);
	}

	_data = *p.get_data();

	return *this;
}

/**********************************************/
/*                                            */
/*                 NN_ListHead                */
/*                                            */
/**********************************************/

template <class _T>
NN_List<_T>::NN_ListHead::NN_ListHead() :
	NN_ListBase(Attr::HEAD)
{

}

template <class _T>
NN_List<_T>::NN_ListHead::NN_ListHead(NN_ListBase* ptr) :
	NN_ListBase(Attr::HEAD)
{
	_nodes.push_back(ptr);
}

template <class _T>
NN_List<_T>::NN_ListHead::NN_ListHead(const NN_ListBase& p) :
	NN_ListBase(Attr::HEAD)
{
	if (p._attr != Attr::HEAD) {
		ErrorExcept(
			"This argument is not a head"
		);
	}

	const std::vector<NN_ListBase*>& src_nodes = *p.get_nodes();

	if (_nodes.capacity() < src_nodes.size()) _nodes.reserve(src_nodes.size());

	for (const NN_ListBase* p_node : src_nodes) {
		NN_ListBase* new_node = NULL;

		if (p_node) {
			switch (p_node->_attr)
			{
			case Attr::NODE:
				new_node = new NN_ListNode(*p_node);
				break;

			case Attr::HEAD:
				new_node = new NN_ListHead(*p_node);
				break;

			default:
				break;
			}
		}

		_nodes.push_back(new_node);
	}
}

template <class _T>
NN_List<_T>::NN_ListHead::NN_ListHead(NN_ListBase&& p) :
	NN_ListBase(Attr::HEAD)
{
	if (p._attr != Attr::HEAD) {
		ErrorExcept(
			"This argument is not a head"
		);
	}

	_nodes = *p.get_nodes();
	p.set_clear_nodes();
}

template <class _T>
NN_List<_T>::NN_ListHead::NN_ListHead(const std::initializer_list<_T>& list) :
	NN_ListBase(Attr::HEAD)
{
	if (_nodes.capacity() < list.size()) _nodes.reserve(list.size());

	for (const _T& elem : list) _nodes.push_back(new NN_ListNode(elem));
}

template <class _T>
NN_List<_T>::NN_ListHead::~NN_ListHead() {
	for (NN_ListBase* p_node : _nodes) delete p_node;
}

template <class _T>
void NN_List<_T>::NN_ListHead::set_clear_nodes() {
	_nodes.clear();
}

template <class _T>
std::vector<typename NN_List<_T>::NN_ListBase*>* NN_List<_T>::NN_ListHead::get_nodes() {
	return &_nodes;
}

template <class _T>
const std::vector<typename NN_List<_T>::NN_ListBase*>* NN_List<_T>::NN_ListHead::get_nodes() const {
	return &_nodes;
}

template <class _T>
void NN_List<_T>::NN_ListHead::set_reserve(size_t capacity) {
	_nodes.reserve(capacity);
}

template <class _T>
size_t NN_List<_T>::NN_ListHead::get_capacity() {
	return _nodes.capacity();
}

template <class _T>
void NN_List<_T>::NN_ListHead::set_nodes(NN_ListBase* p_node) {
	_nodes.push_back(p_node);
}

template <class _T>
const typename NN_List<_T>::NN_ListBase& NN_List<_T>::NN_ListHead::operator=(const NN_ListBase& p) {
	if (this == &p) return *this;

	if (p._attr != Attr::HEAD) {
		ErrorExcept(
			"This argument is not a head"
		);
	}

	clear();

	const std::vector<NN_ListBase*>& src_nodes = *p.get_nodes();

	if (_nodes.capacity() < src_nodes.size()) _nodes.reserve(src_nodes.size());

	for (const NN_ListBase* p_node : src_nodes) {
		NN_ListBase* new_node = NULL;

		if (p_node) {
			switch (p_node->_attr)
			{
			case Attr::NODE:
				new_node = new NN_ListNode(*p_node);
				break;

			case Attr::HEAD:
				new_node = new NN_ListHead(*p_node);
				break;

			default:
				break;
			}
		}

		_nodes.push_back(new_node);
	}

	return *this;
}

template <class _T>
const typename NN_List<_T>::NN_ListBase& NN_List<_T>::NN_ListHead::operator=(NN_ListBase&& p) {
	if (p._attr != Attr::HEAD) {
		ErrorExcept(
			"This argument is not a head"
		);
	}

	clear();
	_nodes = *p.get_nodes();
	p.set_clear_nodes();

	return *this;
}

template <class _T>
void NN_List<_T>::NN_ListHead::clear() {
	for (NN_ListBase* p_node : _nodes) delete p_node;

	_nodes.clear();
}

/**********************************************/
/*                                            */
/*                  Iterator                  */
/*                                            */
/**********************************************/

template <class _T>
NN_List<_T>::Iterator::Iterator(NN_List& list, typename std::vector<NN_ListBase*>::iterator iter) :
	_frame(list._node),
	_iter(iter),
	_sw(true)
{
	if (list._node) {
		if (list._node->_attr == Attr::HEAD) _sw = false;
	}
}

template <class _T>
NN_List<_T>::Iterator::Iterator(const typename NN_List::Iterator& p) :
	_frame(p._frame),
	_iter(p._iter),
	_sw(p._sw)
{

}

template <class _T>
void NN_List<_T>::Iterator::operator++() {
	++_iter;
	_sw = false;
}

template <class _T>
bool NN_List<_T>::Iterator::operator!=(const typename NN_List::Iterator& p) {
	if (_sw) return true;
	else return _iter != p._iter;
}

template <class _T>
NN_List<_T>& NN_List<_T>::Iterator::operator*() {
	if (_sw) return _frame;
	else return *(*_iter);
}

/**********************************************/
/*                                            */
/*                ConstIterator               */
/*                                            */
/**********************************************/

/**********************************************/
/*                                            */
/*                   NN_List                  */
/*                                            */
/**********************************************/

template <class _T>
NN_List<_T>::NN_List(NN_ListBase* node) :
	_node(node),
	_shared(true)
{

}

template <class _T>
NN_List<_T>::NN_List() :
	_node(NULL),
	_shared(false)
{

}

template <class _T>
NN_List<_T>::NN_List(const _T& val) :
	_node(new NN_ListNode(val)),
	_shared(false)
{

}

template <class _T>
NN_List<_T>::NN_List(_T&& val) :
	_node(new NN_ListNode(val)),
	_shared(false)
{

}

template <class _T>
NN_List<_T>::NN_List(const std::initializer_list<_T>& list) :
	_node(new NN_ListHead(list)),
	_shared(false)
{

}

template <class _T>
NN_List<_T>::NN_List(const std::initializer_list<NN_List>& list) :
	_node(new NN_ListHead()),
	_shared(false)
{
	std::vector<NN_ListBase*>& dst_nodes = *_node->get_nodes();

	if (dst_nodes.capacity() < list.size()) dst_nodes.reserve(list.size());

	for (const NN_List& p_list : list) {
		NN_ListBase* new_node = NULL;

		if (p_list._node) {
			switch (p_list._node->_attr)
			{
			case Attr::NODE:
				new_node = new NN_ListNode(*p_list._node);
				break;

			case Attr::HEAD:
				new_node = new NN_ListHead(*p_list._node);
				break;

			default:
				break;
			}
		}

		dst_nodes.push_back(new_node);
	}
}

template <class _T>
NN_List<_T>::NN_List(const NN_List& p) :
	_node(NULL),
	_shared(false)
{
	if (p._node) {
		const NN_ListBase& p_head = *p._node;
		NN_ListBase* new_node = NULL;

		switch (p_head._attr)
		{
		case Attr::NODE:
			new_node = new NN_ListNode(p_head);
			break;

		case Attr::HEAD:
			new_node = new NN_ListHead(p_head);
			break;

		default:
			break;
		}

		_node->get_nodes()->push_back(new_node);
	}
}

template <class _T>
NN_List<_T>::NN_List(NN_List&& p) :
	_node(p._node),
	_shared(false)
{
	p._node = NULL;
}

template <class _T>
NN_List<_T>::~NN_List() {
	if (!_shared) delete _node;
}

template <class _T>
const typename NN_List<_T>::NN_ListBase* NN_List<_T>::get_node() const {
	return _node;
}

template <class _T>
void NN_List<_T>::append(const _T& val) {
	if (!_node) _node = new NN_ListNode(val);
	else {
		switch (_node->_attr)
		{
		case Attr::NODE:
			_node = new NN_ListHead(_node);
			_node->set_nodes(new NN_ListNode(val));
			break;

		case Attr::HEAD:
			_node->set_nodes(new NN_ListNode(val));
			break;

		default:
			break;
		}
	}
}

template <class _T>
void NN_List<_T>::append(_T&& val) {
	if (!_node) _node = new NN_ListNode(val);
	else {
		switch (_node->_attr)
		{
		case Attr::NODE:
			_node = new NN_ListHead(_node);
			_node->set_nodes(new NN_ListNode(val));
			break;

		case Attr::HEAD:
			_node->set_nodes(new NN_ListNode(val));
			break;

		default:
			break;
		}
	}
}

template <class _T>
void NN_List<_T>::append(const std::initializer_list<_T>& list) {
	if (!_node) _node = new NN_ListHead(list);
	else {
		switch (_node->_attr)
		{
		case Attr::NODE:
			_node = new NN_ListHead(_node);
			_node->set_nodes(new NN_ListHead(list));
			break;

		case Attr::HEAD:
			_node->set_nodes(new NN_ListHead(list));
			break;

		default:
			break;
		}
	}
}

template <class _T>
void NN_List<_T>::append(const NN_List& list) {
	if (!_node) _node = new NN_ListHead(*list._node);
	else {
		switch (_node->_attr)
		{
		case Attr::NODE:
			_node = new NN_ListHead(_node);
			_node->set_nodes(new NN_ListHead(*list._node));
			break;

		case Attr::HEAD:
			_node->set_nodes(new NN_ListHead(*list._node));
			break;

		default:
			break;
		}
	}
}

template <class _T>
void NN_List<_T>::extend(const _T& val) {
	if (!_node) _node = new NN_ListNode(val);
	else {
		switch (_node->_attr)
		{
		case Attr::NODE:
			_node = new NN_ListHead(_node);
			_node->set_nodes(new NN_ListNode(val));
			break;

		case Attr::HEAD:
			_node->set_nodes(new NN_ListNode(val));
			break;

		default:
			break;
		}
	}
}

template <class _T>
void NN_List<_T>::extend(_T&& val) {
	if (!_node) _node = new NN_ListNode(val);
	else {
		switch (_node->_attr)
		{
		case Attr::NODE:
			_node = new NN_ListHead(_node);
			_node->set_nodes(new NN_ListNode(val));
			break;

		case Attr::HEAD:
			_node->set_nodes(new NN_ListNode(val));
			break;

		default:
			break;
		}
	}
}

template <class _T>
void NN_List<_T>::extend(const std::initializer_list<_T>& list) {
	if (!_node) _node = new NN_ListHead(list);
	else {
		if (_node->_attr == Attr::NODE) _node = new NN_ListHead(_node);
	}
	
	for (const _T& elem : list) _node->set_nodes(new NN_ListNode(elem));
}

template <class _T>
void NN_List<_T>::extend(const NN_List& list) {
	if (!_node) _node = new NN_ListHead(*list._node);
	else {
		if (_node->_attr == Attr::NODE) _node = new NN_ListHead(_node);
	}

	for (const NN_List& p_list : list)
}

template <class _T>
_T& NN_List<_T>::val() {

}

template <class _T>
size_t NN_List<_T>::size() {

}

template <class _T>
void NN_List<_T>::reserve(size_t size) {

}

template <class _T>
void NN_List<_T>::resize(size_t size) {

}

template <class _T>
bool NN_List<_T>::is_scalar() {

}

template <class _T>
bool NN_List<_T>::is_empty() {

}

template <class _T>
_T& NN_List<_T>::val() const {

}

template <class _T>
size_t NN_List<_T>::size() const {

}

template <class _T>
bool NN_List<_T>::is_scalar() const {

}

template <class _T>
bool NN_List<_T>::is_empty() const {

}


/**********************************************/
/*                                            */
/*					   misc                   */
/*                                            */
/**********************************************/

#ifdef PUT_LIST
template <class _T>
std::ostream& operator<<(std::ostream& os, const NN_List<_T>& list) {
	if (list.get_node()) return list.get_node()->put_list(os) << std::endl;
	else return os << "[]" << std::endl;
}

#endif