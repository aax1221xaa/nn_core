#pragma once
#include "../cpp_source/nn_list.h"
#include "../cpp_source/nn_tensor.h"
#include "../cpp_source/gpu_tensor.h"



/**********************************************/
/*                                            */
/*                NN_Optimizer                */
/*                                            */
/**********************************************/

class NN_Optimizer {
public:
	NN_Optimizer();
	virtual ~NN_Optimizer();

	virtual NN_Optimizer* create(const std::vector<GpuTensor<nn_type>>& weights) const;
	virtual void run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient);
};


/**********************************************/
/*                                            */
/*					   SGD                    */
/*                                            */
/**********************************************/

class SGD : public NN_Optimizer {
	std::vector<GpuTensor<nn_type>> _weights;
	std::vector<GpuTensor<nn_type>> _moments;

	float _l_rate;
	float _m_rate;

	SGD(const std::vector<GpuTensor<nn_type>> weights);

public:
	SGD(float l_rate, float m_rate);
	NN_Optimizer* create(const std::vector<GpuTensor<nn_type>>& weights) const;
	void run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient);
};


/**********************************************/
/*                                            */
/*					 RmsProp                  */
/*                                            */
/**********************************************/

class RmsProp : public NN_Optimizer {
	std::vector<GpuTensor<nn_type>> _weights;
	std::vector<GpuTensor<nn_type>> _square_g;

	float _d_rate;
	float _l_rate;

	RmsProp(const std::vector<GpuTensor<nn_type>> weights);

public:
	RmsProp(float d_rate, float l_rate);
	NN_Optimizer* create(const std::vector<GpuTensor<nn_type>>& weights) const;
	void run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient);
};


/**********************************************/
/*                                            */
/*					   Adam                   */
/*                                            */
/**********************************************/

class Adam : public NN_Optimizer {
	std::vector<GpuTensor<nn_type>> _weights;
	std::vector<GpuTensor<nn_type>> _square_g;
	std::vector<GpuTensor<nn_type>> _decay_g;

	float _l_rate;
	float _beta1;
	float _beta2;

	Adam(const std::vector<GpuTensor<nn_type>> weights);

public:
	Adam(float l_rate, float beta1, float beta2);
	NN_Optimizer* create(const std::vector<GpuTensor<nn_type>>& weights) const;
	void run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient);
};