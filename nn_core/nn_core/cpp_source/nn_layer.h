#pragma once
#include "nn_base.h"


/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

class NN_Dense : public NN_Layer {
public:
	GpuMat _weight;
	GpuMat _bias;
	const int _amounts;

	NN_Dense(const int amounts, const char* name);

	void get_output_shape(const std::vector<nn_shape>& input_shape, std::vector<nn_shape>& output_shape);
	void build(const std::vector<nn_shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuMat>& input, std::vector<GpuMat>& output);
};


/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

class NN_ReLU : public NN_Layer {
public:
	NN_ReLU(const char* name);

	void get_output_shape(const std::vector<nn_shape>& input_shape, std::vector<nn_shape>& output_shape);
	void build(const std::vector<nn_shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuMat>& input, std::vector<GpuMat>& output);
};


/**********************************************/
/*                                            */
/*                   NN_Flat                  */
/*                                            */
/**********************************************/

class NN_Flat : public NN_Layer {
public:
	NN_Flat(const char* name);

	void get_output_shape(const std::vector<nn_shape>& input_shape, std::vector<nn_shape>& output_shape);
	void build(const std::vector<nn_shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuMat>& input, std::vector<GpuMat>& output);
};


/**********************************************/
/*                                            */
/*                 NN_Conv2D                  */
/*                                            */
/**********************************************/

class NN_Conv2D : public NN_Layer {
public:
	const int _amounts;
	const nn_shape _filter_size;
	const nn_shape _stride;
	const Pad _pad;

	GpuMat _filter;
	GpuMat _bias;

	NN_Conv2D(int amounts, const nn_shape& filter_size, const nn_shape& stride, Pad pad, const char* name);

	void get_output_shape(const std::vector<nn_shape>& input_shape, std::vector<nn_shape>& output_shape);
	void build(const std::vector<nn_shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuMat>& input, std::vector<GpuMat>& output);
};


/**********************************************/
/*                                            */
/*                NN_Maxpool2D                */
/*                                            */
/**********************************************/

class NN_Maxpool2D : public NN_Layer {
public:
	const Pad _pad;

	const nn_shape _k_size;
	const nn_shape _stride;

	GpuMat _indice;

	NN_Maxpool2D(const nn_shape& k_size, const nn_shape& stride, const Pad pad, const char* name);

	void get_output_shape(const std::vector<nn_shape>& input_shape, std::vector<nn_shape>& output_shape);
	void build(const std::vector<nn_shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuMat>& input, std::vector<GpuMat>& output);
};