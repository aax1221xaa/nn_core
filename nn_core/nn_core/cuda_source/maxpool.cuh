#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

#include "../cpp_source/misc.h"


/**********************************************

				  _maxpool2d

**********************************************/

class maxpool2dParam {
public:
	int _kh;
	int _kw;
	int _stride_h;
	int _stride_w;

	maxpool2dParam();
	maxpool2dParam(int kh, int kw, int stride_h, int stride_w);
	void set(int kh, int kw, int stride_h, int stride_w);
	const bool is_valid() const;
};

class _maxpool2d : public solutionBase {
public:
	const tensor4d& _input;
	tensor4d _output;

	const maxpool2dParam& _param;

	_maxpool2d(const tensor4d& input, const maxpool2dParam& param);
	const tensor4d calculate_size();
	void operator()(cudaStream_t* s, const nn_type* input, nn_type* output);
};

/**********************************************

				  _dMaxpool2d

**********************************************/

class _dMaxpool2d : public solutionBase {
public:
	const tensor4d& _d_output;
	const _maxpool2d& _maxpool;

	_dMaxpool2d(const tensor4d& d_output, const _maxpool2d& maxpool);
	const tensor4d calculate_size();
	void operator()(const nn_type* d_output, nn_type* d_input);
};

#endif // !MAXPOOL_CUH
