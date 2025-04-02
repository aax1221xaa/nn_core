#pragma once

#include <iostream>
#include <vector>

struct NN_Shape2D {
	int _n;
	int _c;
};

struct NN_Shape4D {
	int _n;
	int _h;
	int _w;
	int _c;
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
	typedef std::vector<int>::const_iterator c_iter;
	typedef std::vector<int>::iterator iter;
	typedef std::vector<int>::reverse_iterator r_iter;
	typedef std::vector<int>::const_reverse_iterator cr_iter;

	NN_Shape();
	NN_Shape(size_t len);
	NN_Shape(const std::initializer_list<int>& list);
	NN_Shape(const int* dims, int ranks);
	NN_Shape(const NN_Shape& p);
	NN_Shape(NN_Shape&& p);

	NN_Shape& operator=(const NN_Shape& p);
	NN_Shape& operator=(NN_Shape&& p);
	int& operator[](int index);
	const int& operator[](int index) const;
	bool operator!=(const NN_Shape& shape) const;

	int ranks();
	int ranks() const;
	size_t total_size();
	size_t total_size() const;
	std::ostream& put_shape(std::ostream& os);
	std::ostream& put_shape(std::ostream& os) const;
	bool is_empty();
	bool is_empty() const;
	void clear();

	iter begin();
	iter end();
	r_iter rbegin();
	r_iter rend();

	c_iter begin() const;
	c_iter end() const;
	cr_iter rbegin() const;
	cr_iter rend() const;

	void push_front(int n);
	void push_front(const NN_Shape& p);
	void push_front(const std::initializer_list<int>& list);

	void push_back(int n);
	void push_back(const NN_Shape& p);
	void push_back(const std::initializer_list<int>& list);

	std::vector<int>& get_dims();
	const std::vector<int>& get_dims() const;

	NN_Shape4D get_4dims();
	NN_Shape2D get_2dims();

	NN_Shape4D get_4dims() const;
	NN_Shape2D get_2dims() const;
};

const char* shape_to_str(const NN_Shape& shape);
//std::string shape_to_str(const std::vector<int>& shape);

std::ostream& operator<<(std::ostream& os, const NN_Shape& shape);