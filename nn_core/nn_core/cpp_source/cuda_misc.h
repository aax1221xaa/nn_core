#pragma once
#include "../cpp_source/cuda_common.h"


/*******************************************

			    tensor4dParam

*******************************************/

class tensor4d {
public:
	int _n;
	int _c;
	int _h;
	int _w;

	bool _is_valid;

	tensor4d();
	tensor4d(int n, int c, int h, int w);
	void set(int n, int c, int h, int w);
	size_t get_size();

	static const char* shape_to_str(const tensor4d& tensor);
};