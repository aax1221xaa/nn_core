#pragma once
#include "cuda_common.h"



class NN_Shape : public NN_Shared_Ptr {
public:
	int* shape;
	int len;

	class Iterator {
	public:
		int* p_shape;
		int index;

		Iterator(int* _shape, int _index);
		Iterator(const Iterator& p);
		
		const Iterator& operator++();
		const Iterator& operator--();
		bool operator!=(const Iterator& p) const;
		bool operator==(const Iterator& p) const;
		int& operator*() const;
	};

	NN_Shape();
	NN_Shape(const NN_Shape& p);
	NN_Shape(const initializer_list<int>& _shape);
	~NN_Shape();
	
	void set(const initializer_list<int>& _shape);
	void clear();
	const char* get_str() const;

	int& operator[](int axis) const;
	const Iterator begin() const;
	const Iterator end() const;

	bool operator==(const NN_Shape& p);
	const NN_Shape& operator=(const NN_Shape& p);

	const size_t get_elem_size() const;
};

typedef NN_Shape* NN_Shape_t;