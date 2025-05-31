#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "../cpp_source/nn_list.h"
#include "../cpp_source/nn_tensor_plus.h"



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
	SGD(nn_type l_rate, nn_type m_rate);
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
	RmsProp(nn_type d_rate, nn_type l_rate);
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

	nn_type _l_rate;
	nn_type _beta1;
	nn_type _beta2;

	Adam(const std::vector<GpuTensor<nn_type>> weights);

public:
	Adam(nn_type l_rate, nn_type beta1, nn_type beta2);
	NN_Optimizer* create(const std::vector<GpuTensor<nn_type>>& weights) const;
	void run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient);
};

#endif