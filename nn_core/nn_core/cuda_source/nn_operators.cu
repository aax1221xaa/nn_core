#include "nn_operators.cuh"


Add::Add() :
	OperatorBase()
{

}

GpuTensor<nn_type> Add::run(const GpuTensor<nn_type>& a, const GpuTensor<nn_type>& b) {
	Tensor<nn_type> h_a(a.get_shape());
	Tensor<nn_type> h_b(b.get_shape());
	Tensor<nn_type> h_c(a.get_shape());
	GpuTensor<nn_type> d_c(a.get_shape());

	h_a = a;
	h_b = b;
	h_c = h_a + h_b;
	d_c = h_c;

	return d_c;
}

OperatorBase* Add::create_forward() {
	return new Add;
}