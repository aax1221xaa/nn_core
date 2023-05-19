#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cuda_source/cuda_indice.cuh"
#include "../cpp_source/misc.h"


/**********************************************

				  conv2dParam

**********************************************/

enum class Pad { VALID, SAME };

class conv2dParam {
public:
	int _w_stride;
	int _h_stride;
	Pad _mode;

	conv2dParam();
	conv2dParam(int w_stride, int h_stride, Pad mode);
	void set(int w_stride, int h_stride, Pad mode);
	const bool is_valid() const;
};

/**********************************************

				 conv2dSolution

**********************************************/

class conv2dSolution : public solutionBase {
public:
	const tensor4d& _input;
	tensor4d _pad;
	const tensor4d& _kernel;
	tensor4d _output;

	const conv2dParam& _param;

	conv2dSolution(const tensor4d& input, const tensor4d& kernel, const conv2dParam& param);
	const tensor4d calculate_size();
	const size_t get_workspace_size();
	void operator()(cudaStream_t* s, const nn_type* input, const nn_type* kernel, nn_type* output, void* workspace);
};

/**********************************************

				 dConv2dSolution

**********************************************/

class dConv2dSolution : public solutionBase {
public:
	const tensor4d& _d_output;
	tensor4d _d_pad;
	tensor4d _d_input;

	const conv2dSolution& _conv;

	dConv2dSolution(const tensor4d& d_output, const conv2dSolution& conv);
	const tensor4d calculate_size();
	const size_t get_workspace_size();
	void operator()(cudaStream_t* s, const nn_type* d_output, const nn_type* kernel, nn_type* d_input, void* workspace);
};

/**********************************************

		      kernelConv2dSolution

**********************************************/

class kernelConv2dSolution : public solutionBase {
public:
	const dConv2dSolution& _d_conv;

	kernelConv2dSolution(const dConv2dSolution& d_conv);
	const size_t get_workspace_size();
	void operator()(const nn_type* d_output, nn_type* gradient, const nn_type* input, void* workspace);
};

#endif // !_CONVOLUTION_CUH_