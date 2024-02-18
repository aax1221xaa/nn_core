#include "nn_optimizer.h"
//#include "../cuda_source/optimizer.cuh"


NN_Optimizer::NN_Optimizer() :
	_weight(NULL),
	_bias(NULL),
	_this(this)
{
}

NN_Optimizer::~NN_Optimizer() {

}

void NN_Optimizer::set(GpuTensor<nn_type>& weight, GpuTensor<nn_type>& bias) {
	_weight = &weight;
	_bias = &bias;
}

void NN_Optimizer::run(const GpuTensor<nn_type>& w_gradient, const GpuTensor<nn_type>& b_gradient) {
	
}

SGD::SGD(float learn_rate, float momentum_rate) :
	_learn_rate(learn_rate),
	_momentum_rate(momentum_rate)
{
}

NN_Optimizer* SGD::create() {
	return new SGD(*this);
}

void SGD::set(GpuTensor<nn_type>& weight, GpuTensor<nn_type>& bias) {
	NN_Optimizer::set(weight, bias);

	_w_momentum = zeros_like(weight);
	_b_momentum = zeros_like(bias);
}

void SGD::run(const GpuTensor<nn_type>& w_gradient, const GpuTensor<nn_type>& b_gradient) {
	//sgd(w_gradient._data, _w_momentum._data, _weight->_data, _weight->_len, _learn_rate, _momentum_rate);
	//sgd(b_gradient._data, _b_momentum._data, _bias->_data, _bias->_len, _learn_rate, _momentum_rate);
}

RmsProp::RmsProp(float learn_rate, float decay_rate) :
	_learn_rate(learn_rate),
	_decay_rate(decay_rate)
{
}

NN_Optimizer* RmsProp::create() {
	return new RmsProp(*this);
}

void RmsProp::set(GpuTensor<nn_type>& weight, GpuTensor<nn_type>& bias) {
	NN_Optimizer::set(weight, bias);

	_w_square = zeros_like(weight);
	_b_square = zeros_like(bias);
}

void RmsProp::run(const GpuTensor<nn_type>& w_gradient, const GpuTensor<nn_type>& b_gradient) {
	//rms_prop(w_gradient._data, _w_square._data, _weight->_data, _weight->_len, _decay_rate, _learn_rate);
	//rms_prop(b_gradient._data, _b_square._data, _bias->_data, _bias->_len, _decay_rate, _learn_rate);
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

void Adam::set(GpuTensor<nn_type>& weight, GpuTensor<nn_type>& bias) {
	NN_Optimizer::set(weight, bias);

	_w_square = zeros_like(weight);
	_w_decay = zeros_like(weight);

	_b_square = zeros_like(bias);
	_b_decay = zeros_like(bias);
}

void Adam::run(const GpuTensor<nn_type>& w_gradient, const GpuTensor<nn_type>& b_gradient) {
	//adam(w_gradient._data, _w_square._data, _w_decay._data, _weight->_data, _weight->_len, _learn_rate, _beta_1, _beta_2);
	//adam(b_gradient._data, _b_square._data, _b_decay._data, _bias->_data, _bias->_len, _learn_rate, _beta_1, _beta_2);
}