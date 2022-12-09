#pragma once
#include "ObjectID.h"


template <class _T>
class NN_Ptr {
protected:
	static Object_Linker linker;
	Object_ID* id;

	_T* ptr;

	void create();
	void destroy();

public:
	NN_Ptr();
	NN_Ptr(_T* p_object);
	NN_Ptr(const NN_Ptr<_T>& p);
	~NN_Ptr();

	void clear();
	void set(_T* p_object);

	NN_Ptr<_T>& operator=(const NN_Ptr<_T>& p);
	_T* operator->();
	bool operator!=(const NN_Ptr<_T>& p);
	bool operator==(const NN_Ptr<_T>& p);
	_T& operator*();
};


template <class _T>
void NN_Ptr<_T>::create() {
	id = linker.Create();
}

template <class _T>
void NN_Ptr<_T>::destroy() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			delete ptr;
			linker.Erase(id);
		}

		ptr = NULL;
		id = NULL;
	}
}

template <class _T>
Object_Linker NN_Ptr<_T>::linker;

template <class _T>
NN_Ptr<_T>::NN_Ptr() {
	id = NULL;
	ptr = NULL;
}

template <class _T>
NN_Ptr<_T>::NN_Ptr(_T* p_object) {
	if (p_object) {
		create();
		ptr = p_object;
	}
}

template <class _T>
NN_Ptr<_T>::NN_Ptr(const NN_Ptr<_T>& p) {
	id = p.id;
	ptr = p.ptr;

	if (id) {
		++id->ref_cnt;
	}
}

template <class _T>
NN_Ptr<_T>::~NN_Ptr() {
	destroy();
}

template <class _T>
void NN_Ptr<_T>::clear() {
	destroy();
}

template <class _T>
void NN_Ptr<_T>::set(_T* p_object) {
	if (ptr == p_object) return;

	destroy();
	create();

	ptr = p_object;
}

template <class _T>
NN_Ptr<_T>& NN_Ptr<_T>::operator=(const NN_Ptr<_T>& p) {
	if (this == &p) return *this;

	destroy();
	
	id = p.id;
	ptr = p.ptr;

	if (id) {
		++id->ref_cnt;
	}

	return *this;
}

template <class _T>
_T* NN_Ptr<_T>::operator->() {
	return ptr;
}

template <class _T>
bool NN_Ptr<_T>::operator!=(const NN_Ptr<_T>& p) {
	return ptr != p.ptr;
}

template <class _T>
bool NN_Ptr<_T>::operator==(const NN_Ptr<_T>& p) {
	return ptr == p.ptr;
}

template <class _T>
_T& NN_Ptr<_T>::operator*() {
	return *ptr;
}