#pragma once
#include "nn_tensor.h"


class NN_Optimizer {
public:
	DeviceTensor<nn_type>* _weight;
	DeviceTensor<nn_type>* _bias;

	NN_Optimizer* _this;

	NN_Optimizer();
	virtual ~NN_Optimizer();

	virtual NN_Optimizer* create() = 0;
	virtual void set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias);
	virtual void run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient);
};

class SGD : public NN_Optimizer {
public:
	DeviceTensor<nn_type> _w_momentum;
	DeviceTensor<nn_type> _b_momentum;

	float _learn_rate;
	float _momentum_rate;

	SGD(float learn_rate, float momentum_rate);
	NN_Optimizer* create();
	void set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias);
	void run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient);
};

class RmsProp : public NN_Optimizer {
public:
	DeviceTensor<nn_type> _w_square;
	DeviceTensor<nn_type> _b_square;

	float _learn_rate;
	float _decay_rate;

	RmsProp(float learn_rate, float decay_rate);
	NN_Optimizer* create();
	void set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias);
	void run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient);
};

class Adam : public NN_Optimizer {
	DeviceTensor<nn_type> _w_square;
	DeviceTensor<nn_type> _b_square;
	
	DeviceTensor<nn_type> _w_decay;
	DeviceTensor<nn_type> _b_decay;

	float _learn_rate;
	float _beta_1;
	float _beta_2;

	Adam(float learn_rate, float beta_1, float beta_2);
	NN_Optimizer* create();
	void set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias);
	void run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient);
};