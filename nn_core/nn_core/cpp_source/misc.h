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

	tensor4d();
	tensor4d(int n, int c, int h, int w);
	void set(int n, int c, int h, int w);
	const size_t get_size() const;
	const bool is_valid() const;

	static const char* shape_to_str(const tensor4d& tensor);
};

/**********************************************

				  solutionBase

**********************************************/

class solutionBase {
public:
	bool _is_calculated;

	solutionBase();
	
	virtual const tensor4d calculate_size();
	virtual const size_t get_workspace_size();

	void clear_flag();
};