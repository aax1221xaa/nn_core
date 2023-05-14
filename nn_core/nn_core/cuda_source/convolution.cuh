#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cuda_source/cuda_indice.cuh"
#include "../cpp_source/cuda_misc.h"

/**********************************************
											  
				 conv2dSolution 			  

**********************************************/

enum class Pad { VALID, SAME };

class conv2dSolution {
public:
	tensor4d _input;
	tensor4d _pad;
	tensor4d _kernel;
	tensor4d _output;
	
	int _w_stride;
	int _h_stride;

	bool _is_calculated;
	Pad _mode;

	conv2dSolution();
	tensor4d calculate_size(const tensor4d& input, const tensor4d& kernel, int w_stride, int h_stride, Pad mode);
	size_t get_work_size();
	void operator()(cudaStream_t* s, const nn_type* input, nn_type* pad, const nn_type* kernel, nn_type* output);
};

/**********************************************

				 dConv2dSolution

**********************************************/

class dConv2dSolution {
public:
	tensor4d _d_output;
	tensor4d _d_pad;
	tensor4d _d_input;

	conv2dSolution _conv;

	bool _is_calculated;

	dConv2dSolution();
	tensor4d calculate_size(const tensor4d& d_output, conv2dSolution& conv);
	size_t get_work_size();
	void operator()(cudaStream_t* s, const nn_type* d_output, const nn_type* kernel, nn_type* d_input, nn_type* workspace);
};

/**********************************************

		      kernelConv2dSolution

**********************************************/

class kernelConv2dSolution {
public:
	const dConv2dSolution& _d_conv;

	bool _is_calculated;

	kernelConv2dSolution(const dConv2dSolution& d_conv);
	size_t get_work_size();
	void operator()(const nn_type* d_output, nn_type* gradient, const nn_type* input, void* workspace);
};


#endif // !_CONVOLUTION_CUH_