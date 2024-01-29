#include "nn_optimizer.h"
#include "../cuda_source/optimizer.cuh"


NN_Optimizer::NN_Optimizer() :
	_weight(NULL),
	_bias(NULL),
	_this(this)
{
}

NN_Optimizer::~NN_Optimizer() {

}

void NN_Optimizer::set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias) {
	_weight = &weight;
	_bias = &bias;
}

void NN_Optimizer::run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient) {
	
}

SGD::SGD(float learn_rate, float momentum_rate) :
	_learn_rate(learn_rate),
	_momentum_rate(momentum_rate)
{
}

NN_Optimizer* SGD::create() {
	return new SGD(*this);
}

void SGD::set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias) {
	NN_Optimizer::set(weight, bias);

	_w_momentum = DeviceTensor<nn_type>::zeros_like(weight);
	_b_momentum = DeviceTensor<nn_type>::zeros_like(bias);
}

void SGD::run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient) {
	sgd(w_gradient, _w_momentum, *_weight, _learn_rate, _momentum_rate);
	sgd(b_gradient, _b_momentum, *_bias, _learn_rate, _momentum_rate);
}

RmsProp::RmsProp(float learn_rate, float decay_rate) :
	_learn_rate(learn_rate),
	_decay_rate(decay_rate)
{
}

NN_Optimizer* RmsProp::create() {
	return new RmsProp(*this);
}

void RmsProp::set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias) {
	NN_Optimizer::set(weight, bias);

	_w_square = DeviceTensor<nn_type>::zeros_like(weight);
	_b_square = DeviceTensor<nn_type>::zeros_like(bias);
}

void RmsProp::run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient) {
	rms_prop(w_gradient, _w_square, *_weight, _decay_rate, _learn_rate);
	rms_prop(b_gradient, _b_square, *_bias, _decay_rate, _learn_rate);
}

Adam::Adam(float learn_rate, float beta_1, float beta_2) :
	_learn_rate(learn_rate),
	_beta_1(beta_1),
	_beta_2(beta_2)
{
}

NN_Optimizer* Adam::create() {
	return new Adam(*this);
}

void Adam::set(DeviceTensor<nn_type>& weight, DeviceTensor<nn_type>& bias) {
	NN_Optimizer::set(weight, bias);

	_w_square = DeviceTensor<nn_type>::zeros_like(weight);
	_w_decay = DeviceTensor<nn_type>::zeros_like(weight);

	_b_square = DeviceTensor<nn_type>::zeros_like(bias);
	_b_decay = DeviceTensor<nn_type>::zeros_like(bias);
}

void Adam::run(const DeviceTensor<nn_type>& w_gradient, const DeviceTensor<nn_type>& b_gradient) {
	adam(w_gradient, _w_square, _w_decay, *_weight, _learn_rate, _beta_1, _beta_2);
	adam(b_gradient, _b_square, _b_decay, *_bias, _learn_rate, _beta_1, _beta_2);
}