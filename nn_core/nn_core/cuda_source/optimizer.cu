#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "optimizer.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>



/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __sgd(
	float* gradient,
	float* weight,
	float* w_momentum,
	cuint w_len,
	float learn_rate,
	float momentum_rate
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < w_len) {
		float m = momentum_rate * w_momentum[idx] + learn_rate * gradient[idx];
		weight[idx] -= m;
		w_momentum[idx] = m;
	}
}

__global__ void __rms_prop(
	float* gradient,
	float* weight,
	float* g,
	cuint w_len,
	float learn_rate,
	float decay_rate
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < w_len) {
		float grad = gradient[idx];
		float _g = decay_rate * g[idx] + (1 - decay_rate) * __powf(grad, 2.f);
		
		weight[idx] -= learn_rate / __powf(_g + EPSILON, 0.5f) * grad;
		g[idx] = _g;
	}
}

__global__ void __adam(
	float* gradient,
	float* weight,
	float* square_g,
	float* decay_g,
	cuint w_len,
	float learn_rate,
	float beta_1,
	float beta_2
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < w_len) {
		float grad = gradient[idx];
		float m = beta_1 * decay_g[idx] + (1 - beta_1) * grad;
		float v = beta_2 * square_g[idx] + (1 - beta_2) * __powf(grad, 2.f);
		
		float _m = m / (1 - beta_1);
		float _v = v / (1 - beta_2);

		weight[idx] -= learn_rate / (_v + EPSILON) * _m;
		decay_g[idx] = m;
		square_g[idx] = v;
	}
}


/**********************************************/
/*                                            */
/*                NN_Optimizer                */
/*                                            */
/**********************************************/

NN_Optimizer::NN_Optimizer() {

}

NN_Optimizer::~NN_Optimizer() {

}

NN_Optimizer* NN_Optimizer::create(const std::vector<GpuTensor<nn_type>>& weights) {
	ErrorExcept(
		"[NN_Optimizer::create] make this function."
	);

	return NULL;
}

void NN_Optimizer::run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient) {
	ErrorExcept(
		"[NN_Optimizer::run] make this function."
	);
}


/**********************************************/
/*                                            */
/*					   SGD                    */
/*                                            */
/**********************************************/

SGD::SGD(const std::vector<GpuTensor<nn_type>> weights) :
	_weights(weights),
	_l_rate(0.f),
	_m_rate(0.f)
{
	for (const GpuTensor<nn_type>& m_weight : weights) {
		const NN_Shape shape = m_weight.get_shape();

		_moments.push_back(GpuTensor<nn_type>::zeros(shape));
	}
}

SGD::SGD(float l_rate, float m_rate) :
	_l_rate(l_rate),
	_m_rate(m_rate)
{
}

NN_Optimizer* SGD::create(const std::vector<GpuTensor<nn_type>>& weights) {
	SGD* optimizer = new SGD(weights);

	optimizer->_l_rate = _l_rate;
	optimizer->_m_rate = _m_rate;

	return optimizer;
}

void SGD::run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient) {
	dim3 threads(BLOCK_1024);

	for (size_t i = 0; i < _weights.size(); ++i) {
		size_t w_len = _weights[i].get_shape().total_size();
		dim3 blocks = get_grid_size(threads, w_len);

		__sgd<<<blocks, threads>>>(
			gradient[i].get_ptr(),
			_weights[i].get_ptr(),
			_moments[i].get_ptr(),
			w_len,
			_l_rate,
			_m_rate
		);
	}
}


/**********************************************/
/*                                            */
/*					 RmsProp                  */
/*                                            */
/**********************************************/

RmsProp::RmsProp(const std::vector<GpuTensor<nn_type>> weights) :
	_weights(weights),
	_d_rate(0.f),
	_l_rate(0.f)
{
	for (const GpuTensor<nn_type>& m_weight : weights) {
		const NN_Shape shape = m_weight.get_shape();

		_square_g.push_back(GpuTensor<nn_type>::zeros(shape));
	}
}

RmsProp::RmsProp(float d_rate, float l_rate) :
	_d_rate(d_rate),
	_l_rate(l_rate)
{
}

NN_Optimizer* RmsProp::create(const std::vector<GpuTensor<nn_type>>& weights) {
	RmsProp* optimizer = new RmsProp(weights);

	optimizer->_d_rate = _d_rate;
	optimizer->_l_rate = _l_rate;

	return optimizer;
}

void RmsProp::run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient) {
	dim3 blocks(BLOCK_1024);
	
	for (size_t i = 0; i < _weights.size(); ++i) {
		size_t len = _weights[i].get_shape().total_size();
		dim3 threads = get_grid_size(blocks, len);

		__rms_prop<<<blocks, threads>>>(
			gradient[i].get_ptr(),
			_weights[i].get_ptr(),
			_square_g[i].get_ptr(),
			len,
			_l_rate,
			_d_rate
		);
	}
}


/**********************************************/
/*                                            */
/*					   Adam                   */
/*                                            */
/**********************************************/

Adam::Adam(const std::vector<GpuTensor<nn_type>> weights) :
	_weights(weights),
	_l_rate(0.f),
	_beta1(0.f),
	_beta2(0.f)
{
	for (const GpuTensor<nn_type>& m_weight : weights) {
		const NN_Shape shape = m_weight.get_shape();

		_square_g.push_back(GpuTensor<nn_type>::zeros(shape));
		_decay_g.push_back(GpuTensor<nn_type>::zeros(shape));
	}
}

Adam::Adam(float l_rate, float beta1, float beta2) :
	_l_rate(l_rate),
	_beta1(beta1),
	_beta2(beta2)
{
}

NN_Optimizer* Adam::create(const std::vector<GpuTensor<nn_type>>& weights) {
	Adam* optimizer = new Adam(weights);

	optimizer->_l_rate = _l_rate;
	optimizer->_beta1 = _beta1;
	optimizer->_beta2 = _beta2;

	return optimizer;
}

void Adam::run(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& gradient) {
	dim3 blocks(BLOCK_1024);

	for (size_t i = 0; i < _weights.size(); ++i) {
		size_t len = _weights[i].get_shape().total_size();
		dim3 threads = get_grid_size(blocks, len);

		__adam<<<blocks, threads>>>(
			gradient[i].get_ptr(),
			_weights[i].get_ptr(),
			_square_g[i].get_ptr(),
			_decay_g[i].get_ptr(),
			len,
			_l_rate,
			_beta1,
			_beta2
		);
	}
}