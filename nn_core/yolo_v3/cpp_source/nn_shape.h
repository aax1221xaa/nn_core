#pragma once
#include "cuda_common.h"
#include <H5Cpp.h>


/**********************************************/
/*                                            */
/*               NN_Tensor4dShape			  */
/*                                            */
/**********************************************/

struct NN_Tensor4dShape {
	int _n;
	int _h;
	int _w;
	int _c;
};


/**********************************************/
/*                                            */
/*               NN_Filter4dShape			  */
/*                                            */
/**********************************************/

struct NN_Filter4dShape {
	int _h;
	int _w;
	int _in_c;
	int _out_c;
};


/**********************************************/
/*                                            */
/*                   NN_Shape				  */
/*                                            */
/**********************************************/

class NN_Shape {
protected:
	std::vector<int> _dims;

public:
	typedef std::vector<int>::const_iterator ConstIterator;
	typedef std::vector<int>::iterator Iterator;

	NN_Shape();
	NN_Shape(size_t len);
	NN_Shape(size_t len, int val);
	NN_Shape(const int* p_dims, int n_dims);
	NN_Shape(const hsize_t* p_dims, int n_dims);
	NN_Shape(const std::initializer_list<int>& list);
	NN_Shape(const NN_Shape& p);
	NN_Shape(NN_Shape&& p);

	NN_Shape& operator=(const NN_Shape& p);
	NN_Shape& operator=(NN_Shape&& p);
	int& operator[](int index);
	const int& operator[](int index) const;
	bool operator!=(const NN_Shape& shape) const;

	int ranks() const;
	size_t total_size() const;
	std::ostream& put_shape(std::ostream& os) const;
	bool is_empty() const;
	void clear();

	Iterator begin();
	Iterator end();

	ConstIterator begin() const;
	ConstIterator end() const;

	void push_front(int n);
	void push_front(const NN_Shape& p);
	void push_front(const std::initializer_list<int>& list);

	void push_back(int n);
	void push_back(const NN_Shape& p);
	void push_back(const std::initializer_list<int>& list);

	int pop(int index);
	int pop(ConstIterator iter);

	void insert(int index, int n);
	void insert(ConstIterator iter, int n);

	const std::vector<int>& get_dims() const;
	const std::vector<uint> get_udims() const;

	NN_Tensor4dShape get_4d_shape();
	NN_Tensor4dShape get_4d_shape() const;

	NN_Filter4dShape get_filter_shape();
	NN_Filter4dShape get_filter_shape() const;
};

const char* shape_to_str(const NN_Shape& shape);
//std::string shape_to_str(const std::vector<int>& shape);

std::ostream& operator<<(std::ostream& os, const NN_Shape& shape);