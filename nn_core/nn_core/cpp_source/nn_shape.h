#pragma once
#include "cuda_common.h"


class NN_Shape {
private:
	std::vector<int> _dims;

public:
	typedef std::vector<int>::const_iterator c_iterator;
	typedef std::vector<int>::iterator iterator;

	NN_Shape();
	NN_Shape(int len);
	NN_Shape(const std::initializer_list<int>& list);
	NN_Shape(const NN_Shape& p);
	NN_Shape(NN_Shape&& p);

	NN_Shape& operator=(const NN_Shape& p);
	NN_Shape& operator=(NN_Shape&& p);
	int& operator[](int index);
	const int& operator[](int index) const;
	bool operator!=(const NN_Shape& shape) const;

	const int get_len() const;
	size_t total_size() const;
	std::ostream& put_shape(std::ostream& os) const;
	bool is_empty() const;
	void clear();

	std::vector<int>::iterator begin();
	std::vector<int>::iterator end();

	std::vector<int>::const_iterator begin() const;
	std::vector<int>::const_iterator end() const;

	void push_front(int n);
	void push_front(const NN_Shape& p);
	void push_front(const std::initializer_list<int>& list);

	void push_back(int n);
	void push_back(const NN_Shape& p);
	void push_back(const std::initializer_list<int>& list);
};

const char* shape_to_str(const NN_Shape& shape);
std::ostream& operator<<(std::ostream& os, const NN_Shape& shape);