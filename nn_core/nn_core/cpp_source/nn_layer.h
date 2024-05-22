#pragma once
#include "../cuda_source/matmul.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/cuda_misc.cuh"
#include "../cuda_source/cuda_indice.cuh"
#include "../cuda_source/convolution.cuh"
#include "../cuda_source/maxpool.cuh"








/**********************************************/
/*                                            */
/*                   NN_Flat                  */
/*                                            */
/**********************************************/

class NN_Flat : public NN_Layer {
public:
	NN_Flat(const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};


