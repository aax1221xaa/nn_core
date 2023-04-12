#pragma once
#include "nn_manager.h"


#if !(FIX_MODE)

/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	vector<int> _shape;

	NN_Input(const vector<int>& input_size, int batch, const char* _layer_name);
	~NN_Input();

	shape_type calculate_output_size(shape_type& input_shape);
	void build(shape_type& input_shape);
	NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input);
	NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output);
};

Layer_t Input(const vector<int>& input_size, int batch, const char* layer_name = "");

/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/

class NN_Test : public NN_Layer {
public:
	NN_Test(const char* name);
	NN_Test(const NN_Test& p);
	
	shape_type calculate_output_size(shape_type& input_shape);
	void build(shape_type& input_shape);
	NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input);
	NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output);
};

/*
class NN_Dense : public NN_Layer {
public:
	NN_Tensor weight;
	NN_Tensor bias;
	int amounts;

	NN_Dense(int _amounts, const string& _layer_name);

	vector<int> calculate_output_size(vector<int>& input_shape);
	void build(vector<int>& input_shape);
	NN_Tensor run(vector<NN_Tensor>& input);
};
*/

#else

/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	vector<int> _shape;

	NN_Input(const vector<int>& input_size, int batch, const char* _layer_name);
	~NN_Input();

	shape_type calculate_output_size(shape_type& input_shape);
	void build(shape_type& input_shape);
	NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input);
	NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output);
};

Layer_t Input(const vector<int>& input_size, int batch, const char* layer_name = "");

/**********************************************/
/*                                            */
/*                   NN_Test                  */
/*                                            */
/**********************************************/

class NN_Test : public NN_Layer {
public:
	NN_Test(const char* name);
	NN_Test(const NN_Test& p);

	shape_type calculate_output_size(shape_type& input_shape);
	void build(shape_type& input_shape);
	NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input);
	NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output);
};

/*
class NN_Dense : public NN_Layer {
public:
	NN_Tensor weight;
	NN_Tensor bias;
	int amounts;

	NN_Dense(int _amounts, const string& _layer_name);

	vector<int> calculate_output_size(vector<int>& input_shape);
	void build(vector<int>& input_shape);
	NN_Tensor run(vector<NN_Tensor>& input);
};
*/

#endif