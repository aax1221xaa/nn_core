#ifndef DENSE_CUH
#define DENSE_CUH

#include "../cpp_source/misc.h"


/**********************************************

				  denseSolution

**********************************************/

class denseSolution : public solutionBase {
public:
	const tensor4d& _input;
	const tensor4d& _weight;
	tensor4d _output;

	denseSolution(const tensor4d& input, const tensor4d& weight);
	const tensor4d calculate_size();
	void operator()(const nn_type* input, const nn_type* weight, nn_type* output);
};

/**********************************************

				  dDenseSolution

**********************************************/

class dDenseSolution : public solutionBase {
public:
	const tensor4d& _d_output;
	const denseSolution& _dense;

	dDenseSolution(const tensor4d& d_output, const denseSolution& dense);
	const tensor4d calcuiate_size();
	const size_t get_workspace_size();
	void operator()(const nn_type* d_output, const nn_type* weight, nn_type* d_input, void* workspace);
};

/**********************************************

				  wDenseSolution

**********************************************/

class wDenseSolution : public solutionBase {
public:
	const dDenseSolution& _d_dense;

	wDenseSolution(const dDenseSolution& d_dense);
	const size_t get_workspace_size();
	void operator()(const nn_type* d_output, const nn_type* input, nn_type* gradient, void* workspace);
};

#endif