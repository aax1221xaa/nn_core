#pragma once
#include <vector>
#include <memory>
#include "cuda_common.h"


using namespace std;


class NN_Shape {
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
	NN_Shape(const initializer_list<int>& _shape);
	~NN_Shape();

	void set(const initializer_list<int>& _shape);
	void clear();

	int& operator[](int axis);
	const Iterator begin() const;
	const Iterator end() const;

	bool operator==(const NN_Shape& p);
};

typedef shared_ptr<NN_Shape> NN_Shape_t;

const char* shape_to_str(const NN_Shape& shape);